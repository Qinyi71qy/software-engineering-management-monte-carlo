#!/usr/bin/env python3
"""基于 Beta-PERT 与蒙特卡洛模拟的软件项目工期风险评估。

输入：data/tasks.csv，包含 id、name、preds、a、m、b 六列；也可用 --input 指定。
输出：results/ 中的 JSON、CSV、NPZ 样本，figures/ 中的七张 PNG 图。
模块：输入与拓扑检查、工期抽样与全网排期、统计及独立种子稳定性分析、绘图。
运行：python src/simulate.py --n 100000 --seed 20261121 --deadline 60
工期单位为工作日。基准模型无资源限制，活动工期相互独立。
"""
import argparse, csv, json, platform, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

ROOT=Path(__file__).resolve().parents[1]

def load_tasks(path):
    if not path.is_file():
        raise FileNotFoundError(
            f'找不到任务数据：{path}。请保留提交包中的 data/tasks.csv，'
            '或使用 --input 指定同结构 CSV 文件。'
        )
    with open(path,encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError('任务 ID 重复')
    done=set(); tasks=[]
    while len(tasks)<len(rows):
        ready=[r for r in rows if r['id'] not in done and set(filter(None,r['preds'].split(';')))<=done]
        if not ready: raise ValueError('前置任务不存在或存在环路')
        for r in ready:
            t=dict(r); t['preds']=list(filter(None,t['preds'].split(';')))
            for k in ['a','m','b']: t[k]=float(t[k])
            if not 0<=t['a']<=t['m']<=t['b']: raise ValueError('必须满足 0 <= a <= m <= b')
            tasks.append(t); done.add(t['id'])
    return tasks

def graph(tasks):
    ix={t['id']:i for i,t in enumerate(tasks)}
    pre=[[ix[p] for p in t['preds']] for t in tasks]; suc=[[] for _ in tasks]
    for j,ps in enumerate(pre):
        for p in ps: suc[p].append(j)
    return pre,suc

def schedule(d,pre,suc,back=False):
    """每行一个情景。合流处取前驱 EF 的最大值，不能把并行任务相加。"""
    d=np.atleast_2d(d); es=np.zeros_like(d); ef=np.zeros_like(d)
    for j,ps in enumerate(pre):
        if ps: es[:,j]=ef[:,ps].max(axis=1)
        ef[:,j]=es[:,j]+d[:,j]
    T=ef.max(axis=1)
    if not back: return T,es,ef
    lf=np.zeros_like(d); ls=np.zeros_like(d)
    for j in range(d.shape[1]-1,-1,-1):
        lf[:,j]=ls[:,suc[j]].min(axis=1) if suc[j] else T
        ls[:,j]=lf[:,j]-d[:,j]
    return T,es,ef,ls,lf

def sample(tasks,n,seed,lam=4,kind='pert'):
    rng=np.random.default_rng(seed)
    # 按情景生成 n x k 数组，保证前缀样本随 n 改变保持一致。
    u=rng.random((n,len(tasks))) if kind=='triangular' else None
    a=np.array([t['a'] for t in tasks]); m=np.array([t['m'] for t in tasks]); b=np.array([t['b'] for t in tasks])
    w=b-a; safe=np.where(w==0,1,w)
    alpha=1+lam*(m-a)/safe; beta=1+lam*(b-m)/safe
    if kind=='pert': return a+w*rng.beta(alpha,beta,size=(n,len(tasks)))
    cut=(m-a)/safe
    return np.where(u<cut,a+np.sqrt(u*w*(m-a)),b-np.sqrt((1-u)*w*(b-m)))

def wilson(p,n):
    z=1.959963984540054; den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(mid-half),float(mid+half)]

def stats(T,deadline):
    n=len(T); p=float(np.mean(T<=deadline)); se=float(np.std(T,ddof=1)/np.sqrt(n))
    return dict(n=n,mean=float(T.mean()),std=float(T.std(ddof=1)),mean_ci95=[float(T.mean()-1.96*se),float(T.mean()+1.96*se)],
        p=float(p),p_ci95=wilson(p,n),min=float(T.min()),max=float(T.max()),
        quantiles={str(q):float(np.quantile(T,q)) for q in [.05,.1,.5,.8,.9,.95,.99]})

def ranks(x):
    """带并列值的平均秩，避免使用仅适合无并列值的简化 Spearman 公式。"""
    order=np.argsort(x,kind='mergesort'); xs=x[order]
    starts=np.r_[0,np.flatnonzero(np.diff(xs))+1]; ends=np.r_[starts[1:],len(x)]
    rank=np.empty(len(x),float)
    rank[order]=np.repeat((starts+ends-1)/2+1,ends-starts)
    return rank

def write_csv(path,rows):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

# 图形模块仅依赖 Pillow。英文轴标签保证缺少中文字体的电脑也能复现。
COL={'blue':'#2B5C88','red':'#B04C42','teal':'#2E8074','gray':'#627080','light':'#E2E8EE','ink':'#172B3A'}
def font(size):
    candidates=[ROOT/'assets'/'DejaVuSans.ttf',Path('/System/Library/Fonts/Supplemental/Arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),Path('C:/Windows/Fonts/arial.ttf')]
    for p in candidates:
        if p.exists(): return ImageFont.truetype(str(p),size)
    # Pillow 10+ 接受 size；较旧版本不接受该参数，退回其内置字体仍可绘图。
    try: return ImageFont.load_default(size=size)
    except TypeError: return ImageFont.load_default()

def canvas(title,subtitle='',height=900):
    im=Image.new('RGB',(1600,height),'white');g=ImageDraw.Draw(im)
    g.text((95,36),title,font=font(34),fill=COL['ink']);g.text((95,88),subtitle,font=font(23),fill=COL['gray'])
    return im,g

def axes(g,x0=150,y0=170,x1=1480,y1=760,xlabel='Duration (working days)',ylabel=''):
    g.line([(x0,y0),(x0,y1),(x1,y1)],fill=COL['gray'],width=3)
    g.text(((x0+x1)//2-120,y1+80),xlabel,font=font(25),fill=COL['ink'])
    if ylabel:g.text((x0,y0-38),ylabel,font=font(23),fill=COL['gray'])

def chart_hist(T,deadline,cpm,out):
    im,g=canvas('Project duration distribution',f'{len(T):,} simulations | Beta-PERT | Y-axis = simulation count')
    counts,edges=np.histogram(T,bins=45); x0,y0,x1,y1=150,170,1480,760
    lo=np.floor(min(cpm,T.min())/5)*5; hi=np.ceil(max(deadline,T.max())/5)*5
    X=lambda x:x0+(x-lo)/(hi-lo)*(x1-x0); Y=lambda y:y1-y/counts.max()*(y1-y0)
    axes(g)
    for c,a,b in zip(counts,edges[:-1],edges[1:]):g.rectangle((X(a),Y(c),X(b)-1,y1),fill=COL['blue'])
    for tick in np.arange(lo,hi+1,5):g.text((X(tick)-15,y1+15),f'{tick:.0f}',font=font(23),fill=COL['gray'])
    for tick in np.linspace(0,counts.max(),5):
        g.text((35,Y(tick)-12),f'{tick:.0f}',font=font(21),fill=COL['gray'])
    for k,(v,label,color) in enumerate([(cpm,'CPM',COL['gray']),(T.mean(),'Mean',COL['teal']),(deadline,'Deadline',COL['red'])]):
        g.line((X(v),y0,X(v),y1),fill=color,width=4)
        g.text((100+k*470,125),f'{label}: {v:.2f} days',font=font(24),fill=color)
    im.save(out)

def chart_cdf(T,deadline,out):
    im,g=canvas('Completion probability curve','Read the delivery promise from a chosen confidence level')
    x=np.sort(T); lo=np.floor(x[0]/5)*5;hi=np.ceil(x[-1]/5)*5
    X=lambda t:150+(t-lo)/(hi-lo)*1330;Y=lambda p:760-p*570
    axes(g)
    for v in np.arange(lo,hi+1,5):g.text((X(v)-15,780),f'{v:.0f}',font=font(22),fill=COL['gray'])
    for p in [0,.2,.4,.6,.8,1]:
        y=Y(p);g.line((150,y,1480,y),fill=COL['light'],width=1);g.text((55,y-14),f'{p:.0%}',font=font(23),fill=COL['gray'])
    idx=np.linspace(0,len(x)-1,min(1800,len(x))).astype(int)
    g.line([(X(x[i]),Y((i+1)/len(x))) for i in idx],fill=COL['blue'],width=5)
    for k,q in enumerate([.5,.8,.9,.95]):
        v=np.quantile(T,q);xx,yy=X(v),Y(q);g.ellipse((xx-7,yy-7,xx+7,yy+7),fill=COL['red'])
        g.text((160+k*350,130),f'P{int(q*100)} = {v:.2f}',font=font(24),fill=COL['red'])
    im.save(out)

def chart_bars(rows,out,key,title,xlabel,percent=False):
    rows=sorted(rows,key=lambda r:r[key],reverse=True); im,g=canvas(title,'Task IDs are defined in the WBS table',height=1100)
    mx=max(r[key] for r in rows)*1.15; x0,x1=210,1430
    for i,r in enumerate(rows):
        y=160+i*51;xx=x0+max(0,r[key])/mx*(x1-x0)
        g.text((110,y+1),r['id'],font=font(27),fill=COL['ink'])
        g.rectangle((x0,y,xx,y+31),fill=COL['red'] if i<3 else COL['blue'])
        v=f"{r[key]:.1%}" if percent else f"{r[key]:.3f}"
        g.text((xx+12,y),v,font=font(24),fill=COL['ink'])
    g.text((470,1025),xlabel,font=font(26),fill=COL['ink']);im.save(out)

def chart_network(tasks,out):
    im,g=canvas('Task dependency network','Finish-to-start relations | all durations are working days',height=1050)
    pos={'A':(95,500),'B':(250,350),'C':(250,650),'D':(400,280),'E':(550,430),'F':(550,190),'G':(720,270),'H':(720,650),'I':(720,470),'J':(890,440),'K':(650,880),'L':(1060,300),'M':(1060,600),'N':(1220,440),'O':(1380,440),'P':(1450,750)}
    bw,bh=112,60
    for t in tasks:
        x,y=pos[t['id']]
        for p in t['preds']:
            px,py=pos[p]
            if p=='A' and t['id']=='K':
                g.line([(px,py+35),(px,880),(x-65,880)],fill='#A3B1BE',width=3)
                g.polygon([(x-65,880),(x-79,873),(x-79,887)],fill='#A3B1BE')
                continue
            dx=x-px;dy=y-py;ll=np.hypot(dx,dy); sx=px+dx/ll*63;sy=py+dy/ll*35;ex=x-dx/ll*63;ey=y-dy/ll*35
            g.line((sx,sy,ex,ey),fill='#A3B1BE',width=3)
            ux,uy=dx/ll,dy/ll
            g.polygon([(ex,ey),(ex-14*ux+7*uy,ey-14*uy-7*ux),(ex-14*ux-7*uy,ey-14*uy+7*ux)],fill='#A3B1BE')
    for t in tasks:
        x,y=pos[t['id']];g.rounded_rectangle((x-bw/2,y-bh/2,x+bw/2,y+bh/2),radius=8,fill='#EAF0F6',outline=COL['blue'],width=3)
        g.text((x-37,y-14),f"{t['id']}  {t['m']:g}d",font=font(23),fill=COL['ink'])
    g.text((100,970),'Arrows show precedence only; positions do not represent a time scale.',font=font(24),fill=COL['gray'])
    im.save(out)

def chart_gantt(tasks,es,ef,critical,out):
    im,g=canvas('CPM baseline schedule','Most-likely durations | resource availability is assumed sufficient',height=1100)
    hi=np.ceil(ef.max()/5)*5;X=lambda x:200+x/hi*1220
    for v in np.arange(0,hi+1,5):
        g.line((X(v),150,X(v),990),fill=COL['light'],width=2);g.text((X(v)-10,1000),f'{v:g}',font=font(22),fill=COL['gray'])
    for i,t in enumerate(tasks):
        y=160+i*50;g.text((100,y),t['id'],font=font(26),fill=COL['ink']);g.rectangle((X(es[i]),y,X(ef[i]),y+30),fill=COL['red'] if critical[i] else COL['blue'])
    g.text((430,1050),'Working day from project start | red = zero total float',font=font(24),fill=COL['ink']);im.save(out)

def chart_scenarios(rows,out):
    im,g=canvas('Scenario comparison','P80 and mean duration | stress assumptions are not observed frequencies')
    maxv=max(r['quantiles']['0.8'] for r in rows)*1.12;X=lambda v:470+v/maxv*900
    for i,r in enumerate(rows):
        y=180+i*92;g.text((95,y+8),r['name'],font=font(26),fill=COL['ink'])
        g.rectangle((470,y,X(r['mean']),y+27),fill=COL['blue']);g.rectangle((470,y+32,X(r['quantiles']['0.8']),y+59),fill=COL['teal'])
        g.text((X(r['quantiles']['0.8'])+12,y+29),f"{r['quantiles']['0.8']:.2f}",font=font(24),fill=COL['ink'])
    g.text((490,820),'Blue = mean    Green = P80    Unit = working days',font=font(26),fill=COL['ink']);im.save(out)

def check_model(input_path):
    # 检查串联、并联汇合、浮动以及确定性参数的退化行为。
    pre=[[],[0],[0],[1,2]];suc=[[1,2],[3],[3],[]]
    r=schedule(np.array([[2.,3.,5.,1.]]),pre,suc,True)
    assert r[0][0]==8 and r[1][0,3]==7
    assert np.allclose(r[3][0]-r[1][0],[0,2,0,0])
    t=[dict(a=3.,m=3.,b=3.)]
    assert np.all(sample(t,100,1)==3)
    try: load_tasks(input_path)
    except Exception as e: raise AssertionError(str(e))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=100000);ap.add_argument('--seed',type=int,default=20261121);ap.add_argument('--deadline',type=float,default=60);ap.add_argument('--input',type=Path,default=ROOT/'data'/'tasks.csv');ap.add_argument('--out',type=Path,default=ROOT);ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
    check_model(args.input)
    if args.check_only: print('Model checks passed');return
    if args.n<10000: ap.error('课程实验要求至少 10000 次模拟')
    out=args.out; (out/'results').mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(parents=True,exist_ok=True)
    tasks=load_tasks(args.input);pre,suc=graph(tasks);n=args.n
    d=sample(tasks,n,args.seed); T,es,ef,ls,lf=schedule(d,pre,suc,True)
    # 全情景约束检查：采样区间、工期与拓扑约束、正向/逆向时差。
    for j,t in enumerate(tasks):
        assert np.all((d[:,j]>=t['a'])&(d[:,j]<=t['b']))
        for p in pre[j]: assert np.all(es[:,j]>=ef[:,p]-1e-10)
    assert np.all(ls-es>=-1e-9)
    m=np.array([t['m'] for t in tasks]);mu=np.array([(t['a']+4*t['m']+t['b'])/6 for t in tasks])
    cp=schedule(m,pre,suc,True);pm=schedule(mu,pre,suc,True)
    cpmrows=[];sens=[]; rankT=ranks(T)
    for j,t in enumerate(tasks):
        w=t['b']-t['a'];al=1+4*(t['m']-t['a'])/w if w else 1;be=1+4*(t['b']-t['m'])/w if w else 1
        cpmrows.append(dict(id=t['id'],name=t['name'],mean=mu[j],alpha=al,beta=be,variance=w*w*al*be/((al+be)**2*(al+be+1)),ES=cp[1][0,j],EF=cp[2][0,j],LS=cp[3][0,j],LF=cp[4][0,j],TF=cp[3][0,j]-cp[1][0,j]))
        fixed=d.copy();fixed[:,j]=mu[j];tf=schedule(fixed,pre,suc)[0]
        rho=float(np.corrcoef(ranks(d[:,j]),rankT)[0,1]) if np.std(d[:,j])>0 else 0.
        sens.append(dict(id=t['id'],name=t['name'],spearman=rho,criticality=float(np.mean(abs(ls[:,j]-es[:,j])<1e-8)),variance_reduction=float(1-tf.var(ddof=1)/T.var(ddof=1))))
    # 路径频率：完整枚举源到汇路径，独立核验最长路径等于项目总工期。
    paths=[]
    def visit(path):
        if not suc[path[-1]]: paths.append(path);return
        for k in suc[path[-1]]:visit(path+[k])
    for j in range(len(tasks)):
        if not pre[j]:visit([j])
    pt=np.column_stack([d[:,p].sum(axis=1) for p in paths]); assert np.allclose(pt.max(axis=1),T)
    chosen=pt.argmax(axis=1);pathrows=[]
    for k,p in enumerate(paths): pathrows.append(dict(path='-'.join(tasks[j]['id'] for j in p),m_duration=float(m[p].sum()),mean_duration=float(mu[p].sum()),critical_frequency=float(np.mean(chosen==k))))
    # 教学情景假设，不是历史数据或已验证的干预效果：对最敏感的三项
    # 活动保持乐观工期 a 不变，将悲观跨度 (b-a) 缩短 30%，并将最可能
    # 工期 m 缩短 1 工作日。实际项目可尝试通过提前澄清审批规则、接口
    # Mock、代码评审和自动回归降低 G、N、F 等任务的返工风险；只有当
    # 执行记录支持这些参数改善时，才应将其作为新的预测输入。
    top=sorted(sens,key=lambda x:x['spearman'],reverse=True)[:3];targets={r['id'] for r in top}
    improved=[dict(t) for t in tasks]
    for t in improved:
        if t['id'] in targets:
            t['b']=t['a']+.7*(t['b']-t['a']);t['m']=min(t['b'],max(t['a'],t['m']-1))
    mitigated=sample(improved,n,args.seed);Ti=schedule(mitigated,pre,suc)[0]
    scenarios=[]
    def add(name,x):scenarios.append(dict(name=name,**stats(x,args.deadline)))
    add('Baseline',T);add('Targeted mitigation',Ti)
    # 共同事件冲击：20% 情景中 F G J L N 同时延长 20%；超出基准上界是设计使然。
    rng=np.random.default_rng(args.seed+1);event=rng.random(n)<.2;stress=d.copy()
    for j,t in enumerate(tasks):
        if t['id'] in {'F','G','J','L','N'}:stress[:,j]*=np.where(event,1.2,1.)
    add('Common-risk stress',schedule(stress,pre,suc)[0])
    add('PERT lambda 2',schedule(sample(tasks,n,args.seed,lam=2),pre,suc)[0])
    add('PERT lambda 8',schedule(sample(tasks,n,args.seed,lam=8),pre,suc)[0])
    add('Triangular',schedule(sample(tasks,n,args.seed,kind='triangular'),pre,suc)[0])
    # 每个规模使用二十个互不相同的随机种子独立抽样；统计重复间波动。
    # 这用于检查蒙特卡洛估计的数值稳定性，不证明模型预测外部有效。
    convergence=[]; convergence_replicates=[]
    for index,k in enumerate([1000,10000,50000,100000]):
        if k>n: continue
        runs=[]
        for rep in range(20):
            # 最大规模把正式基准运行也作为二十次重复中的一次，便于直接对照。
            independent_seed=args.seed if (index==3 and rep==0) else args.seed+1000+index*100+rep
            independent_total=schedule(sample(tasks,k,independent_seed),pre,suc)[0]
            row=dict(size=k,seed=independent_seed,**stats(independent_total,args.deadline))
            convergence_replicates.append(row);runs.append(row)
        means=np.array([r['mean'] for r in runs]);p80=np.array([r['quantiles']['0.8'] for r in runs]);probs=np.array([r['p'] for r in runs])
        convergence.append(dict(size=k,repetitions=len(runs),mean=float(means.mean()),mean_sd=float(means.std(ddof=1)),
            mean_min=float(means.min()),mean_max=float(means.max()),P80=float(p80.mean()),P80_min=float(p80.min()),P80_max=float(p80.max()),
            p=float(probs.mean()),p_min=float(probs.min()),p_max=float(probs.max())))
    repeats=[dict(seed=s,**stats(schedule(sample(tasks,10000,s),pre,suc)[0],args.deadline)) for s in [20261121,20261122,20261123,20261124,20261125]]
    deadlines=[]
    for x in sorted(set([float(cp[0][0]),50.,55.,60.,65.,float(np.ceil(np.quantile(T,.8))),float(np.ceil(np.quantile(T,.9)))])):
        p=float(np.mean(T<=x));deadlines.append(dict(deadline=x,p=p,ci_low=wilson(p,n)[0],ci_high=wilson(p,n)[1]))
    summary=dict(config=dict(n=n,seed=args.seed,deadline=args.deadline,lambda_=4,python=platform.python_version(),numpy=np.__version__,pillow=pillow_version),baseline=stats(T,args.deadline),cpm_m=float(cp[0][0]),cpm_mean=float(pm[0][0]),scenario_p55=[dict(name=r['name'],p55=float(np.mean(x<=55))) for r,x in zip(scenarios[:3],[T,Ti,schedule(stress,pre,suc)[0]])],deadlines=deadlines,sensitivity=sens,paths=pathrows,scenarios=scenarios,convergence=convergence,convergence_replicates=convergence_replicates,repeats=repeats,mitigation_targets=sorted(targets),mitigation_tasks=improved,checks='all passed')
    (out/'results'/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    write_csv(out/'results'/'cpm.csv',cpmrows);write_csv(out/'results'/'sensitivity.csv',sens);write_csv(out/'results'/'path_frequency.csv',pathrows);write_csv(out/'results'/'deadline_probability.csv',deadlines)
    write_csv(out/'results'/'scenario_comparison.csv',[dict(name=r['name'],mean=r['mean'],std=r['std'],P80=r['quantiles']['0.8'],P90=r['quantiles']['0.9'],p_deadline=r['p']) for r in scenarios])
    write_csv(out/'results'/'convergence.csv',convergence)
    write_csv(out/'results'/'convergence_replicates.csv',[dict(size=r['size'],seed=r['seed'],mean=r['mean'],P80=r['quantiles']['0.8'],p=r['p']) for r in convergence_replicates])
    write_csv(out/'results'/'seed_repeats.csv',[dict(seed=r['seed'],mean=r['mean'],P80=r['quantiles']['0.8'],p=r['p']) for r in repeats])
    np.savez_compressed(out/'results'/'baseline_samples.npz',durations=d,total=T,ids=np.array([t['id'] for t in tasks]))
    np.savetxt(out/'results'/'project_duration.csv',T,delimiter=',',header='working_days',comments='',fmt='%.10f')
    chart_network(tasks,out/'figures'/'fig01_network.png');chart_gantt(tasks,cp[1][0],cp[2][0],abs(cp[3][0]-cp[1][0])<1e-8,out/'figures'/'fig02_cpm_gantt.png')
    chart_hist(T,args.deadline,cp[0][0],out/'figures'/'fig03_histogram.png');chart_cdf(T,args.deadline,out/'figures'/'fig04_cdf.png')
    chart_bars(sens,out/'figures'/'fig05_sensitivity.png','spearman','Task duration sensitivity','Spearman rank correlation with project duration')
    chart_bars(sens,out/'figures'/'fig06_criticality.png','criticality','Task criticality index','Fraction of simulations with zero total float',True)
    chart_scenarios(scenarios,out/'figures'/'fig07_scenarios.png')
    print(json.dumps({k:summary[k] for k in ['config','baseline','cpm_m','cpm_mean','mitigation_targets']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
