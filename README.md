# 软件工程管理课程作业 题目一

项目：校园实验室设备预约管理系统工期风险评估。

## 获取源码

源码与报告同包提供。解压本提交包后打开 `src/simulate.py`，任务数据在
`data/tasks.csv`。获取和运行本地提交包不需要账号、访问令牌或私有仓库权限。
若另行发布到 GitHub，仓库链接应与报告附录 B 保持一致，并确认教师能够访问。

## 文件目录

- `report/`：PDF报告与可编辑Word报告。
- `src/simulate.py`：全部仿真、统计、校验和绘图实现，含中文注释。
- `data/tasks.csv`：16项任务、名称、前置依赖、三点工期。
- `results/`：已运行的基准结果、CSV统计表、原始样本。
- `figures/`：全部7张图，可由源码重建。
- `requirements.txt`：运行依赖。
- `VALIDATION.txt`：复跑与模型校验记录。

## 环境和运行

参考环境：Python 3.12.14，NumPy 2.3.5，Pillow 13.0.0。
建议使用 Python 3.12，终端切换到解压目录。

```bash
python -m venv .venv
```

Windows激活：`.venv\Scripts\activate`
macOS/Linux激活：`source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
python src/simulate.py --check-only
python src/simulate.py --n 100000 --seed 20261121 --deadline 60 --out reproduced
```

输出写入 `reproduced/results/` 与 `reproduced/figures/`，保留原交付结果不变。
运行不需要联网，首次安装依赖需要访问包下载源。
请保留 `data/tasks.csv`；若移动或更名，请使用 `--input` 指定其位置。程序
在数据文件缺失时给出明确错误，不会悄悄替换成与报告不一致的内置数据。
如不需要单独目录，删除 `--out reproduced` 会覆盖默认 results 和 figures 文件。
不建议把 --out 指向其他作业目录，以免覆盖同名输出。

## 参数

- `--n`：模拟次数，默认100000，正式实验至少10000。
- `--seed`：基准随机种子，默认20261121。
- `--deadline`：用于按期概率统计的期限，单位工作日，默认60。
- `--input`：另一个同结构CSV文件，默认data/tasks.csv。
- `--out`：输出目录，默认提交包根目录。
- `--check-only`：仅运行内置模型小例校验。

CSV字段：id、name、preds、a、m、b。多个前置ID用分号分隔。
代码拓扑排序后计算；不要求输入CSV预先按拓扑序排列。
原始CSV行顺序影响同层任务排序，可能改变固定seed下的具体随机结果。
同一seed不能保证跨NumPy版本逐项一致；保留版本和原始样本便于审计。

## 结果核对

基准：N=100000，seed=20261121，lambda=4。

- CPM最可能工期：46.0000000000日
- 均值工期输入的CPM：50.3333333333日
- 模拟期望：50.8983346104日
- P80：54.0741024558日
- P90：55.8145300797日
- 60日按期概率：0.99016
- 敏感度前三：G、N、F

图01网络、图02甘特、图03直方图、图04累计概率、图05敏感度、图06关键度、图07情景。
Pillow绘图保证不需要安装Matplotlib；图形统计内容满足题目要求。
为避免中文字体缺失，图内轴标签使用英文和任务ID，报告图题及WBS给出完整中文解释。
代码优先查找系统Arial/DejaVu Sans；无此字体则回退Pillow默认字体。

## 原始样本读取

```python
import numpy as np
x = np.load('results/baseline_samples.npz')
print(x['ids'])                 # 工期矩阵列对应的拓扑任务ID
print(x['durations'].shape)     # (100000, 16)
print(x['total'].mean())
```

`project_duration.csv`提供同一批100000个项目工期。
`summary.json`包含主要统计量、Wilson区间、全部情景、重复实验、版本信息。
`convergence.csv`汇总四种样本规模下各20次独立随机种子的均值、波动和范围；
`convergence_replicates.csv`保留80次原始重复结果。它们用于检查数值稳定性，
不构成模型预测真实项目的有效性证明。
`scenario_comparison.csv`为六个情景汇总；55日方案比较位于summary.json的scenario_p55。

## 模型假设及教学数据

案例工期为透明构造的教学估算，不是企业实测或专家访谈结果。
单位是工作日，采用完成后开始且无时滞的固定依赖；工期基准独立；资源假设足够。
基准PERT端点为硬边界。共同事件压力情景允许超出基准端点。
三点估算改变、前置关系改变、资源不足或范围变更时必须重新建模。

三项缓解：F、G、N的m各减1日（不低于a），b改为a+0.7*(b-a)。
共同风险：20%情景中F、G、J、L、N同时乘1.2。
这些情景参数为实验假设，未证明实际投入能达到该效果。
lambda=2/8改变分布均值和方差；三角分布也改变均值，不是仅改变尾部。
同seed的不同形状Beta采样不构成严格配对随机数实验。

课程要求个人独立完成。提交前填写封面，理解并自行核对修改内容，按教师规定披露辅助工具使用。
如教师指定真实项目或格式，请按要求替换案例并重新计算；修改脚本结果不会自动更新报告正文。
