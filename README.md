# 软件工程管理课程作业：题目一代码

案例：校园实验室设备预约管理系统工期风险评估。这里仅公开复现代码和教学任务数据；Word/PDF 报告不上传 GitHub。

## 文件

- `src/simulate.py`：PERT/Beta 蒙特卡洛模拟、CPM、敏感度、关键度、缓解情景和绘图。
- `data/tasks.csv`：16 项教学估算任务及依赖。
- `requirements.txt`：Python 依赖。

## 运行

参考环境：Python 3.12、NumPy 2.3.5、Pillow 13.0.0。在仓库页面选择 **Code → Download ZIP**，解压并进入仓库根目录，运行：

```bash
python -m pip install -r requirements.txt
python src/simulate.py --check-only
python src/simulate.py --n 100000 --seed 20261121 --deadline 60 --out reproduced
```

结果保存在 `reproduced/results/`，7 张图保存在 `reproduced/figures/`。`data/tasks.csv` 必须与脚本一起保留；使用其他任务文件时传入 `--input`。

## 核对值与范围

基准：CPM 46.000 日；10 万次模拟均值 50.898 日；P80 54.074 日；P90 55.815 日；60 日按期概率 99.016%。独立随机种子稳定性分析、情景比较与样本会在运行时生成。案例参数是教学估算，不代表真实项目测量。
