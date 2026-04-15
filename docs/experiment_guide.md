# 实验操作手册

> 从第一次启动实验到做完整论文级对比，你会重复用到的操作都在这里。
> 配合 `framework_intro.md` 一起读效果更好。

## 目录

1. [环境与依赖安装](#1-环境与依赖安装)
2. [实验生命周期](#2-实验生命周期)
3. [配置文件字段全解](#3-配置文件字段全解)
4. [常见实验变体](#4-常见实验变体)
5. [训练监控与可视化](#5-训练监控与可视化)
6. [Checkpoint 与断点续训](#6-checkpoint-与断点续训)
7. [评估与结果汇总](#7-评估与结果汇总)
8. [调试与常见报错](#8-调试与常见报错)
9. [性能调优](#9-性能调优)
10. [做消融实验的标准流程](#10-做消融实验的标准流程)

---

## 1. 环境与依赖安装

### 1.1 基础环境

```bash
conda create -n tibetan_ss python=3.10 -y
conda activate tibetan_ss

cd "/Users/liuhao/research/speech separation/tibetan_ss"
pip install -e .                  # 把 src/tibetan_ss 挂到 PYTHONPATH
pip install -r requirements.txt
```

### 1.2 按需加装模型依赖

```bash
# MossFormer2 / Dual-path Mamba 都依赖 SpeechBrain
pip install speechbrain

# Dual-path Mamba 额外依赖（CUDA 环境，GPU 机器）
pip install mamba-ssm causal-conv1d

# TIGER
pip install rotary_embedding_torch asteroid-filterbanks

# DIP / Proposed / SepReformer / Identity：不需要额外
```

### 1.3 环境变量

把下列三条写进 `.env`（用 `set -a; source .env; set +a` 加载，或放到 shell rc 里）：

```bash
export TIBETAN_ROOT=/your/path/NICT-Tib1
export DEMAND_ROOT=/your/path/DEMAND
export TIBETAN_SS_OUTPUT=/your/path/tibetan_ss_output   # 存生成的混合数据 + manifests
```

### 1.4 自检

```bash
pytest tests/ -q
```

若 7 个测试全部 PASS，框架本身没问题，剩下的是数据和对标模型依赖。

---

## 2. 实验生命周期

一个完整实验走 4 步：**prepare → train → evaluate → aggregate**。

### 2.1 Prepare (一次性)

```bash
scripts/prepare_data.sh configs/data/sr16k.yaml
```

做了两件事：

1. 扫描 NICT-Tib1 与 DEMAND，按 seed 切分说话人与噪声文件，写到 `${TIBETAN_SS_OUTPUT}/manifests/`
2. 根据 `configs/data/default.yaml::offline.num_mixtures` 离线预生成混合音频到 `${TIBETAN_SS_OUTPUT}/mixtures/{train,val,test}/`

耗时主要卡在 I/O；单机 SSD + 8 核 CPU 生成 20k 训练混合大约 30-60 min。

> **Tip**：首次跑如果不确定 speaker 正则是否匹配 NICT-Tib1 的文件命名，可以加 `--speaker-regex` 覆盖。打开 `speakers_train.json` 抽查几个条目，确认 `gender` 字段是 M/F 而不是 U。

### 2.2 Train (每个实验一次)

```bash
# 单个实验
scripts/train.sh configs/experiment/proposed.yaml

# 加 override
scripts/train.sh configs/experiment/baseline_tiger.yaml \
    training.trainer.max_epochs=150 \
    training.dataloader.batch_size=8
```

### 2.3 Evaluate (通常自动在 fit 结束时触发)

`trainer.fit` 结束后会自动 `trainer.test(ckpt_path="best")`，不用额外做。如果你临时有另一个 ckpt 要评估：

```bash
scripts/evaluate.sh configs/experiment/proposed.yaml \
    outputs/logs/proposed/checkpoints/epoch040-sisdri13.420.ckpt \
    --output outputs/logs/proposed/alt_metrics.json
```

### 2.4 Aggregate (所有实验跑完后)

```bash
python -m tibetan_ss.cli.aggregate_results \
    --root outputs/logs \
    --output outputs/summary.md
```

得到一张 Markdown 表，一目了然：

```
| model                    | SI-SDR | SI-SDRi | PESQ-WB | STOI  |
| ------------------------ | ------ | ------- | ------- | ----- |
| baseline_tiger           | 12.47  | 12.42   | 2.81    | 0.872 |
| baseline_sepreformer     | 13.85  | 13.80   | 3.02    | 0.891 |
| baseline_dual_path_mamba | 13.12  | 13.07   | 2.95    | 0.884 |
| baseline_mossformer2     | 14.20  | 14.15   | 3.08    | 0.897 |
| ext_dip                  | 12.80  | 12.75   | 2.84    | 0.874 |
| proposed                 | 14.40  | 14.35   | 3.12    | 0.901 |
```

### 2.5 一键跑全流程

```bash
scripts/run_all.sh
```

按顺序训练 6 个实验并生成汇总表。通常在单卡 A100 上约 2-3 天，视 epoch 数和数据量而定。

---

## 3. 配置文件字段全解

### 3.1 `configs/data/default.yaml`

```yaml
paths:
  tibetan_root:  # NICT-Tib1 根目录（可用 ${oc.env:TIBETAN_ROOT} 从环境变量注入）
  noise_root:    # DEMAND 根目录
  output_root:   # 输出目录（混合音频 + manifests）

speaker_split:   # PPT 规划：8M+12F → 6+1+1 / 8+2+2
  train: {male: 6, female: 8}
  val:   {male: 1, female: 2}
  test:  {male: 1, female: 2}
  seed: 20260415

sample_rate: 16000     # 16 kHz 默认；换成 8000 对应 sr8k.yaml
num_channels: 1

segment:
  train: 3.0           # 每段长度基准（秒）
  val:   3.0
  test:  null          # null = 不裁剪，用完整 utterance
  random_length: true  # train/val 是否随机在 [min_seconds, max_seconds] 范围采样
  min_seconds: 2.0
  max_seconds: 4.0

mixing:
  num_speakers: 2
  overlap:
    train: {mode: uniform, low: 0.0, high: 1.0}
    val:   {mode: mixture, components: [...]}     # 混合分布
    test:  {mode: mixture, components: [...]}
  level_diff_db: {low: -5.0, high: 5.0}            # K 的采样范围
  snr_db:
    train: {low: 2.5, high: 30.0}
    val:   {low: 2.5, high: 30.0}
    test:  {low: 0.0, high: 30.0}
  gender_pairing: random              # random | same | cross
  normalize: rms                      # 归一化方式
  rms_target_dbfs: -25.0              # 归一化到的参考电平

noise:
  enabled: true
  prob_apply: 1.0                     # 每条 mixture 有多少概率叠加噪声

reverb:
  enabled: false                      # 默认关闭；开启后会在混合前卷积 RIR
  rir_root: null
  t60_range: [0.2, 0.8]

offline:
  num_mixtures: {train: 20000, val: 3000, test: 3000}
  output_subdir: mixtures
  manifest_subdir: manifests
  audio_format: wav
  audio_subtype: PCM_16               # PCM_16 / FLOAT
  seed: 20260415

dynamic_mixing:
  enabled: false
  cache_per_epoch: 20000              # 每 epoch 在线采样这么多 mixture
```

### 3.2 `configs/training/default.yaml`

```yaml
trainer:
  max_epochs: 200
  precision: "16-mixed"               # bf16-mixed / 32-true / 16-mixed
  accelerator: auto                   # gpu / cpu / mps / auto
  devices: auto
  strategy: auto                      # ddp / auto / ddp_find_unused_parameters_true
  gradient_clip_val: 5.0              # 只对 SeparationModule 生效；GAN 用手动 clip
  accumulate_grad_batches: 1
  log_every_n_steps: 50
  deterministic: false

optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-3
  betas: [0.9, 0.999]
  weight_decay: 1.0e-6

scheduler:
  name: cosine                        # cosine / none
  warmup_epochs: 5
  min_lr_ratio: 0.01

dataloader:
  batch_size: 4
  num_workers: 4
  persistent_workers: true
  pin_memory: true

loss:
  main: sisdr_pit                     # 当前只有这一种
  aux_weights: {}                     # 留作扩展

eval_metrics: [si_sdr, si_sdri, pesq_wb, stoi]

checkpoint:
  monitor: val/si_sdri
  mode: max
  save_top_k: 3
  save_last: true

early_stop:
  enabled: true
  monitor: val/si_sdri
  mode: max
  patience: 20

logger:
  name: tensorboard                   # tensorboard / csv
  save_dir: outputs/logs
```

### 3.3 `configs/model/proposed.yaml`（详解）

```yaml
name: proposed

# 编码器（Conv + TCN × 3）
n_filters: 512                        # 滤波器数 N
kernel_size: 32                       # 滤波器长度（samples）；16 kHz 推荐 32，8 kHz 推荐 16
stride: null                          # null = kernel_size // 2
bottleneck: 128                       # TCN 通道数 B
tcn_hidden: 512                       # 中间隐藏通道 H
encoder_tcn_blocks: 8                 # 每个 stack 的 block 数；dilation 1..128
encoder_tcn_repeats: 3                # "TCN × 3" 的 3

# 分支头
branch_tcn_blocks: 8
branch_tcn_repeats: 4                 # "TCN × 4"
perturbation_std: 1.0e-3              # 破对称的高斯扰动强度

# 解码器
decoder_tcn_blocks: 8
decoder_tcn_repeats: 3                # "TCN × 3"
mask_nonlinear: relu                  # relu / sigmoid

# 判别器（只在 engine: gan 下生效）
discriminator:
  n_ffts: [512, 1024, 2048]
  hop_ratio: 0.25
  channels: [32, 64, 128, 256, 512]

# 三阶段训练调度
schedule:
  rep_from_epoch: 10                  # 第 10 个 epoch 开启 L_rep
  gan_from_epoch: 25                  # 第 25 个 epoch 开启 L_D
  rep_weight: 0.05                    # L_rep 的权重
  gan_weight: 0.1                     # generator hinge 的权重

disc_lr: 5.0e-4                       # 判别器学习率（通常为 G 的 0.5x）
```

### 3.4 `configs/experiment/*.yaml`

这层只做"粘合剂"：选 data preset + model preset + training preset，加上实验级元数据：

```yaml
defaults:
  - /data:     sr16k
  - /model:    proposed
  - /training: default
  - _self_
tag: proposed
engine: gan                           # standard | gan
seed: 20260415
```

---

## 4. 常见实验变体

### 4.1 切换采样率

改 experiment YAML 的 `defaults.0.data`：

```yaml
defaults:
  - /data: sr8k       # 原来是 sr16k
  - /model: dual_path_mamba
  - /training: default
  - _self_
```

或命令行临时覆盖一次：

```bash
scripts/train.sh configs/experiment/baseline_tiger.yaml \
    data.sample_rate=8000
```

但注意：模型内部的 `kernel_size/win/stride` 可能是为 16 kHz 设的，直接换 SR 不一定最优，通常要配合模型超参一起调。

### 4.2 开启 Dynamic Mixing

```bash
scripts/train.sh configs/experiment/proposed.yaml \
    data.dynamic_mixing.enabled=true \
    data.dynamic_mixing.cache_per_epoch=30000
```

train 会切到在线混合，val/test 仍走离线 manifest 保持可比。

### 4.3 改 overlap 分布

假设你想让训练只用 overlap ∈ [0.5, 1.0]：

```bash
scripts/train.sh configs/experiment/proposed.yaml \
    data.mixing.overlap.train.low=0.5
```

或者切成混合分布：直接编辑 `configs/data/default.yaml::mixing.overlap.train` 为 `{mode: mixture, components: [...]}`。

### 4.4 开启混响

把 `configs/data/default.yaml::reverb.enabled=true`，并指向一个装满 `.wav` RIR 文件的目录：

```yaml
reverb:
  enabled: true
  rir_root: /data/rirs
  t60_range: [0.2, 0.8]
```

⚠️ **当前实现状态**：混合脚本里的 `MixtureSimulator` 保留了 reverb hook 但还没接上真实的 RIR 卷积；启用前需要先在 `src/tibetan_ss/data/mixing.py` 里按照 `reverb.enabled` 分支加一段 `scipy.signal.fftconvolve(src, rir)`。TODO 详见 `mixing.py` 顶部注释。

### 4.5 切换损失

目前 `SeparationModule` 硬编码了 PIT + SI-SDR。若想换 PIT + SI-SNR 或加入 MRSTFT：

1. 在 `src/tibetan_ss/losses/` 新建 `mrstft.py`
2. 在 `SeparationModule._step` 里按 `training.loss.main` 分支调用
3. 在 `configs/training/default.yaml::loss.main` 支持新名字

> PIT 的核心（`pit.py::_pairwise_loss`）与具体 pair 函数解耦——只要你的 loss 接收 `(est, ref) → scalar`，就能塞进 PIT。

### 4.6 同时跑多种模型做 sweep

```bash
for exp in baseline_tiger baseline_sepreformer proposed; do
    scripts/train.sh configs/experiment/${exp}.yaml \
        training.trainer.max_epochs=80 \
        tag=${exp}_sweep_short
done
```

用 `tag=xxx` override 避免覆盖先前的 log 目录。

---

## 5. 训练监控与可视化

### 5.1 TensorBoard

```bash
tensorboard --logdir outputs/logs --port 6006
```

标签约定：

```
train/loss            总训练 loss
train/loss_main       只记录 L_main
train/loss_rep        (GAN 模式) L_rep
train/loss_g          generator hinge
train/loss_d          discriminator hinge
val/loss
val/si_sdr / si_sdri / pesq_wb / stoi
test/si_sdr / si_sdri / pesq_wb / stoi
lr-AdamW              optimizer 0 的学习率（GAN 模式下 generator）
```

### 5.2 CSV Logger

把 `training.logger.name` 改成 `csv`，metrics 会写到 `outputs/logs/<tag>/csv/version_0/metrics.csv`，方便 pandas 读入做自定义画图。

### 5.3 `resolved.yaml`

每个实验启动时会在 `outputs/logs/<tag>/resolved.yaml` 写下**最终生效的**配置（含 CLI override），复现时直接用这份即可。

---

## 6. Checkpoint 与断点续训

### 6.1 保存策略

- 按 `val/si_sdri` top-3 保留
- `last.ckpt` 每个 epoch 都会更新
- 文件名形如 `epoch040-sisdri13.420.ckpt`

### 6.2 从中断处恢复

```bash
python -m tibetan_ss.cli.train --config configs/experiment/proposed.yaml \
    +trainer.ckpt_path=outputs/logs/proposed/checkpoints/last.ckpt
```

（当前 CLI 没做 `--resume` 的 shortcut，需要手工传 trainer.ckpt_path。）

### 6.3 加载 ckpt 做推理（脱离 CLI）

```python
from omegaconf import OmegaConf
from tibetan_ss.engine import SeparationModule
from tibetan_ss.models import build_model

cfg = OmegaConf.to_container(OmegaConf.load("outputs/logs/proposed/resolved.yaml"), resolve=True)
model = build_model({**cfg["model"], "sample_rate": cfg["data"]["sample_rate"]})
pl_module = SeparationModule.load_from_checkpoint(
    "outputs/logs/proposed/checkpoints/last.ckpt",
    model=model, training_cfg=cfg["training"],
)
pl_module.eval()

import torchaudio
mix, sr = torchaudio.load("some_mixture.wav")           # (1, T)
est = pl_module(mix)                                     # (1, 2, T)
torchaudio.save("s1.wav", est[0, 0:1], sr)
torchaudio.save("s2.wav", est[0, 1:2], sr)
```

---

## 7. 评估与结果汇总

### 7.1 评估单个 ckpt

```bash
scripts/evaluate.sh configs/experiment/proposed.yaml \
    outputs/logs/proposed/checkpoints/last.ckpt \
    --output eval.json
```

### 7.2 自定义指标子集

```bash
scripts/evaluate.sh configs/experiment/proposed.yaml \
    outputs/logs/proposed/checkpoints/last.ckpt \
    training.eval_metrics=["si_sdr","si_sdri","estoi"]
```

支持的指标：`si_sdr`, `si_sdri`, `pesq_wb`, `pesq_nb`, `stoi`, `estoi`。

### 7.3 聚合多实验

```bash
python -m tibetan_ss.cli.aggregate_results \
    --root outputs/logs \
    --output outputs/summary.md
```

这里只读 CSV logger 的 `metrics.csv`；如果你只用了 TensorBoard，需要先补一个 CSV logger 再跑。最简单的办法：训练时把 `training.logger.name` 设为 `csv`，或者同时启用两个 logger（需要改 `cli/train.py::_build_logger`）。

---

## 8. 调试与常见报错

### 8.1 OOM / CUDA out of memory

- 调小 `training.dataloader.batch_size`
- 切到 `precision: 16-mixed`（AMP）
- 缩短 `data.segment.train` 长度
- 用 `accumulate_grad_batches` 模拟大 batch

### 8.2 `AttributeError: 'NoneType' object has no attribute 'backward'`

你在 `ProposedGANModule` 里改代码时，某条分支返回了 `None` 作为 loss。手动优化模式下每个 `opt.step()` 前必须有 `manual_backward(loss)`。

### 8.3 `RuntimeError: Expected all tensors to be on the same device`

Lightning 2.x 下用 `pl_module.device` 而不是 `next(pl_module.parameters()).device`。一般出现在自己写的 metric 里忘了 `.to(self.device)`。

### 8.4 PESQ 全部 NaN

- 检查 `sample_rate`：PESQ-WB 要 16 kHz，PESQ-NB 要 8 kHz
- 检查输入片段长度：`pesq` 要求至少 0.5 s 有效内容
- 查 `val/si_sdri`：如果也是 NaN，八成是 mixture 或 ref 里有 inf

### 8.5 SepReformer ImportError

```
ImportError: cannot import name 'logger_wraps' from 'utils.decorators'
```

SepReformer 里用了相对 import。`_thirdparty_path.register_thirdparty("SepReformer")` 会把它的根目录加到 `sys.path`；若仍出错，执行一次：

```bash
cd third_party/SepReformer && ls utils/decorators.py
```

确认文件存在，再检查 `sys.path` 里是否已经有这条路径。

### 8.6 MossFormer2 `Could not locate MossFormer2`

SpeechBrain 的模型路径在不同版本里不同。请先：

```bash
python -c "import speechbrain; print(speechbrain.__version__)"
python -c "from speechbrain.lobes.models import mossformer"  # 试这几个
```

找到正确路径后加到 `src/tibetan_ss/models/mossformer2.py::_CANDIDATE_PATHS` 列表顶部。

### 8.7 `mamba-ssm` 编译失败

- macOS / Intel Mac：不支持。换一台 Linux GPU 机。
- Linux：`pip install mamba-ssm --no-build-isolation`，或先 `pip install packaging ninja`
- CUDA 版本不匹配：`pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html`

---

## 9. 性能调优

### 9.1 DataLoader

- `num_workers` 设成 CPU 物理核数的一半左右；太多反而会卡 GPU 喂不进
- `persistent_workers=true` 在 workers > 0 时强烈推荐，否则每个 epoch 启动都要重建
- 混合文件存到 **本地 NVMe** 而不是 NFS，否则 I/O 是瓶颈

### 9.2 Precision

- `precision: 16-mixed` 是 AMP，对大多数分离模型无损（SI-SDR 差 <0.05 dB）
- `precision: bf16-mixed` 在 A100 / H100 上比 fp16 更稳定（但 RTX3090/4090 对 bf16 有支持但慢）
- `precision: 32-true` 只在你怀疑数值不稳时用

### 9.3 Gradient Accumulation

batch 小的时候（如 MossFormer2-L 只能放 batch=1 的 30 s segment）:

```bash
scripts/train.sh configs/experiment/baseline_mossformer2.yaml \
    training.trainer.accumulate_grad_batches=8
```

等效 batch=8，但显存只用 batch=1。

### 9.4 选择合适的 chunk 长度

- Mamba / Transformer 是 **O(T)** 或 **O(T log T)**；segment 3 s 已经足够
- TF-domain 模型的 STFT `win/stride` 决定 chunk 下采样倍数；改 `win/stride` 比改 segment 更有效
- 对于 `SepReformer_Large_DM_WHAMR`，推理时 segment=null（整段 5-10 s）比训练时 segment=3 效果好得多

### 9.5 DDP 多卡

想一次用多卡：

```bash
scripts/train.sh configs/experiment/proposed.yaml \
    training.trainer.devices=4 \
    training.trainer.strategy=ddp
```

注意 Lightning 的 DDP 会把 batch_size 平均切到每张卡，所以总 batch = `batch_size × devices`；学习率一般要线性放大。

---

## 10. 做消融实验的标准流程

拿"验证 L_rep 是否有用"为例：

### Step 1 — 复制 proposed.yaml

```bash
cp configs/experiment/proposed.yaml configs/experiment/proposed_no_rep.yaml
```

### Step 2 — 修改 tag 并关闭 L_rep

把 `tag: proposed_no_rep`，然后在 `configs/model/proposed.yaml`（或在 experiment 里 inline override）把 `schedule.rep_from_epoch` 设成 `9999`（永远不启用）。

更干净的做法是用 CLI override（不污染 config）：

```bash
scripts/train.sh configs/experiment/proposed.yaml \
    tag=proposed_no_rep \
    model.schedule.rep_from_epoch=99999
```

### Step 3 — 跑多个 seed 取均值

```bash
for seed in 42 123 2024; do
    scripts/train.sh configs/experiment/proposed.yaml \
        tag=proposed_no_rep_seed${seed} \
        seed=${seed} \
        model.schedule.rep_from_epoch=99999
done
```

### Step 4 — 汇总

```bash
python -m tibetan_ss.cli.aggregate_results --root outputs/logs --output ablation.md
```

### Step 5 — 报告

最终表格里只保留核心对比列：

| 变体 | SI-SDRi | PESQ-WB | 备注 |
|------|---------|---------|------|
| proposed (full)    | 14.35 | 3.12 | 完整三阶段 |
| -L_rep            | 13.98 | 3.05 | 关闭 L_rep |
| -L_D              | 14.10 | 3.02 | 关闭判别器 |
| -L_rep -L_D       | 13.72 | 2.93 | 退化到 PIT+SI-SDR |

> 消融做 3 次 seed 平均 ± std 是最低底线，论文评审会看。
