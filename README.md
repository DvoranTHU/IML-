# README




## 仓库结构

```
大实验/
├── README.md                      # 运行说明
├── requirements.txt               # Python 依赖
│
├── configs/
│   └── default.yaml               # 路径、特征、模型、评估配置
│
├── report/
│   ├── .gitkeep
│   ├── report.md                  # 实验报告
│   └── report.pdf                  # 实验报告
│
├── scripts/                       # 入口脚本
│   ├── build_metadata.py          # 处理数据集
│   ├── validate_audio.py          # 音频质检
│   ├── extract_features.py        # 特征提取
│   ├── train_gmm.py               # GMM-MA
│   ├── train_svm.py               # 核SVM
│   ├── train_gbdt.py              # GBDT
│   ├── train_stacking.py          # Stacking
│   └── run_cross_corpus.py        # 跨语料评估
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # 加载、解析配置
│   │
│   ├── datasets/                  # 处理数据集
│   │   ├── __init__.py
│   │   ├── ravdess.py
│   │   ├── crema_d.py
│   │   └── emodb.py
│   │
│   ├── audio/
│   │   ├── __init__.py            # 音频质检、加载
│   │   ├── load.py                
│   │   └── validate.py
│   │
│   ├── features/                  # 特征提取
│   │   ├── __init__.py
│   │   ├── frame.py               
│   │   ├── aggregate.py           
│   │   └── extract.py             
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gmm_map.py             # GMM分类器
│   │   ├── kernel_svm.py          # SVM分类器
│   │   ├── gbdt.py                # LightGBM实现的GBDT
│   │   └── stacking.py            # Stacking
│   │
│   └── eval/
│       ├── __init__.py            # 评估工具、Split分折、Runner统一训练和预测
│       ├── metrics.py             
│       ├── utils.py               
│       ├── splits.py              
│       └── runner.py              
│
├── data/                          # 原始数据，提供网盘链接
│   ├── RAVDESS/
│   ├── CREMA-D/
│   └── EMODB/
│
└── outputs/                       # 运行输出，提供网盘链接
    ├── .gitkeep
    ├── metadata/
    ├── features/
    ├── metrics/
    └── figures/
```



## 网盘链接

| 资源 | 说明 | 清华云盘链接 |
|------|------|--------------|
| **`data/`** | 原始音频，RAVDESS、CREMA-D、EMODB数据集 | https://cloud.tsinghua.edu.cn/f/0b22a52e345743c9a234/ |
| **`outputs/`** | 特征缓存、评估指标JSON等 | https://cloud.tsinghua.edu.cn/f/bef90efd859548bd8e3d/ |



## 环境配置

Python 3.10

```bash
conda create --name mli_lab
conda activate mli_lab

pip install -r requirements.txt
```



## 运行方法

将数据集放至 data/：

```bash
# 1. 提取数据集元数据、验证音频到元数据
python scripts/build_metadata.py
python scripts/validate_audio.py

# 2. 特征提取
python scripts/extract_features.py

# 3. RAVDESS主实验
python scripts/train_gmm.py
python scripts/train_svm.py
python scripts/train_gbdt.py
python scripts/train_stacking.py

# 4. 跨语料评估
python scripts/run_cross_corpus.py
```

配置统一由`configs/default.yaml`读取，随机种子`seed: 42`，结果输出到`outputs/metrics/*.json`。



## 输出路径

| 阶段 | 输出 |
|------|------|
| 元数据 | `outputs/metadata/dataset.csv` |
| 特征 | `outputs/features/{ravdess,crema_d,emodb}.npz` |
| 主实验指标 | `outputs/metrics/gmm_ravdess.json` |
| | `outputs/metrics/svm_ravdess.json` |
| | `outputs/metrics/gbdt_ravdess.json` |
| | `outputs/metrics/stacking_ravdess.json` |
| 跨语料指标 | `outputs/metrics/cross_corpus_crema_d.json` |
| | `outputs/metrics/cross_corpus_emodb.json` |

主指标为UAR无权重平均召回，报告中的均值±标准差及95%置信区间见各JSON的summary字段。
