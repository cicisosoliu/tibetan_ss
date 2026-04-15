# tibetan_ss — 藏语多人语音分离统一实验框架

> 对标 5 个公开分离模型 + 我们提出的 **Early-Separation** 模型，统一数据管道 / 训练 loop / 评估指标，目标是在 NICT-Tib1 藏语数据上完成公平对比并支撑论文级实验。

---

## 目录结构

```
tibetan_ss/
├── configs/                      # 层叠 YAML（data / model / training / experiment）
│   ├── data/{default,sr16k,sr8k}.yaml
│   ├── model/{dual_path_mamba,mossformer2,tiger,sepreformer,dip_frontend,proposed}.yaml
│   ├── training/default.yaml
│   └── experiment/<6 个一键入口>.yaml
├── docs/
│   ├── data_pipeline.md          # 数据生成 / 说话人划分 / 标签规则
│   ├── models.md                 # 6 个模型的实现说明 + 出处
│   └── reproduce.md              # 从零跑到指标的完整命令
├── scripts/
│   ├── prepare_data.sh           # 一键准备：NICT-Tib1 + DEMAND + 离线混合
│   ├── train.sh                  # 单实验训练
│   ├── evaluate.sh               # 单实验评估
│   └── run_all.sh                # 顺序跑 6 个实验并聚合结果
├── src/tibetan_ss/
│   ├── data/                     # mixing / Dataset / DataModule / 预处理脚本
│   ├── models/                   # base + proposed + 5 个对标适配器
│   ├── engine/                   # Lightning module（标准 + GAN）+ metrics
│   ├── losses/                   # SI-SDR / PIT
│   ├── cli/                      # train / evaluate / aggregate_results
│   └── utils/
├── third_party/                  # 克隆进来的官方仓库（gitignored）
├── tests/                        # smoke tests
├── requirements.txt
└── pyproject.toml
```

---

## 快速开始

### 1. 环境

```bash
conda create -n tibetan_ss python=3.10 -y
conda activate tibetan_ss

cd tibetan_ss
pip install -e .
pip install -r requirements.txt

# 按需加装对标模型所需扩展依赖
pip install speechbrain                           # MossFormer2, Dual-path Mamba
pip install mamba-ssm causal-conv1d               # Dual-path Mamba (CUDA only)
pip install rotary_embedding_torch asteroid-filterbanks  # TIGER
```

### 2. 克隆对标仓库

```bash
cd third_party
git clone --depth 1 https://github.com/xi-j/Mamba-TasNet.git
git clone --depth 1 https://github.com/JusperLee/TIGER.git
git clone --depth 1 https://github.com/dmlguq456/SepReformer.git
# MossFormer2 直接从 speechbrain 导入；DIP 按论文自行复现
cd ..
```

### 3. 数据准备

```bash
export TIBETAN_ROOT=/data/NICT-Tib1
export DEMAND_ROOT=/data/DEMAND
export TIBETAN_SS_OUTPUT=$PWD/data     # 输出目录（≈ 几十 GB）

scripts/prepare_data.sh configs/data/sr16k.yaml
```

### 4. 训练 + 评估

```bash
scripts/train.sh configs/experiment/proposed.yaml
# 或 baseline：
scripts/train.sh configs/experiment/baseline_tiger.yaml

# 评估某个 ckpt：
scripts/evaluate.sh configs/experiment/proposed.yaml \
    outputs/logs/proposed/checkpoints/last.ckpt

# 一键跑 6 个实验并生成汇总表：
scripts/run_all.sh
```

---

## 模型清单

| 模型 | 类型 | 入口 config | 说明文档 |
|------|------|------------|----------|
| Dual-path Mamba (Jiang et al. 2025) | SSM | `baseline_dual_path_mamba.yaml` | [models.md#dual-path-mamba](docs/models.md#dual-path-mamba) |
| MossFormer2 (Zhao et al. 2024) | Transformer + FSMN | `baseline_mossformer2.yaml` | [models.md#mossformer2](docs/models.md#mossformer2) |
| TIGER (Xu et al. 2024) | TF-domain band-split | `baseline_tiger.yaml` | [models.md#tiger](docs/models.md#tiger) |
| SepReformer (Shin et al. 2024) | 非对称 Enc-Dec | `baseline_sepreformer.yaml` | [models.md#sepreformer](docs/models.md#sepreformer) |
| DIP Frontend (Wang et al. 2024) | SSL 前端 + 下游分离 | `ext_dip.yaml` | [models.md#dip-frontend](docs/models.md#dip-frontend) |
| **Proposed: Early-Separation** | 双分支 TCN + 共享解码 + 频谱判别器 | `proposed.yaml` | [models.md#proposed](docs/models.md#proposed) |

---

## 文档导航

必读三件套（新协作者从这里开始）：

1. **框架总览** → [docs/framework_intro.md](docs/framework_intro.md)
   设计哲学、模块职责矩阵、一条训练指令的完整生命周期、核心契约、依赖矩阵
2. **实验操作手册** → [docs/experiment_guide.md](docs/experiment_guide.md)
   生命周期 / 配置字段全解 / 常见变体 / 调试 / 性能调优 / 消融流程
3. **新增模型教程** → [docs/add_new_model.md](docs/add_new_model.md)
   三种集成路径 + 完整模板 + Checklist + 常见坑 + 端到端案例

参考资料：

- 数据管道、说话人划分、标签规则 → [docs/data_pipeline.md](docs/data_pipeline.md)
- 每个模型的架构说明、依赖安装、默认超参 → [docs/models.md](docs/models.md)
- 从零复现实验的命令列表 → [docs/reproduce.md](docs/reproduce.md)

---

## 设计要点

1. **统一接口**：所有模型实现 `forward(mixture: (B, T)) -> (B, 2, T)` 的 `BaseSeparator` 合约，由 `registry` 动态构造，训练/评估/推理 loop 完全复用。
2. **标签可追溯**：离线混合时每条样本生成 `mix_<split>_<gender>_ov<overlap>_n<noise>_ns<snr>_rl<K>_<idx>` 形式的 ID，同时写入 JSON manifest，实验流程可对齐 PPT 第 11 页的标签规则。
3. **动态混合可切换**：`configs/data/default.yaml` 里 `dynamic_mixing.enabled` 切换离线 / 在线混合，无需修改训练代码。
4. **三阶段训练**：提出模型的 `L_main → +L_rep → +L_D` 通过 `configs/model/proposed.yaml::schedule` 控制，`ProposedGANModule` 按 epoch 自动启用。
5. **混响留接口**：`configs/data/default.yaml::reverb.enabled=false` 为默认值；打开后按 `rir_root` 卷积即可，无需改 Dataset。

---

## 已知限制

- MossFormer2 / Dual-path Mamba 的前向结果假设了固定的输出通道顺序。若遇到 dimension mismatch 请在 `configs/model/<model>.yaml` 调整 `n_speakers` 或 `variant`。
- PESQ-WB 仅支持 16 kHz 输入；8 kHz 运行时退回 PESQ-NB。
- `segment.test=null` 会将 test dataloader 的 batch size 强制为 1（变长推理），速度慢但符合原始论文 evaluation 习惯。
