# NICT-Tib1 本地冒烟测试

> 目标：从零把 NICT-Tib1 (已下载到 `tibetan_ss/Tibetan/`) 接入框架，**dynamic mixing 做训练**，用提出模型在 3 个 epoch 内跑完整 train / val / test cycle。

## 0. 前置条件一览

| 项 | 要求 |
|----|------|
| Tibetan 数据 | `tibetan_ss/Tibetan/{data/, wav.scp, label.txt, README}` 已解压 |
| Python | 3.10+ |
| GPU | 可选（CPU 也能跑，约 30 min；GPU 约 5 min） |
| DEMAND 噪声 | **不要求**（冒烟测试默认关闭噪声） |

## 1. 数据集结构速查

```
tibetan_ss/Tibetan/
├── data/
│   ├── 006/006_01/006_*.wav      # 400 wav, 女
│   ├── 007/007_01,...,007_03/    # 1600 wav, 女
│   ├── ...
│   └── 077/077-1/                # 400 wav, 男（注意目录名用了短横线）
├── label.txt                      # Kaldi 格式: utt_id\t transcript
├── wav.scp                        # Kaldi 格式: utt_id\t 相对路径
└── README
```

- **20 位说话人**：3 位数 ID `006, 007, ..., 077`
- **16 kHz / 单声道 / 16-bit PCM**
- **16799 条 utterance** 共 ≈ 33.5 h
- **性别在命名中没有体现** —— 靠我们预先跑 F0 估计出来的 `configs/data/nict_tib1_speakers.yaml` 做映射（8 男 12 女，已验证与 README 一致）

## 2. 安装依赖

```bash
cd "/Users/liuhao/research/speech separation/tibetan_ss"

# 框架本体
pip install -e .
pip install -r requirements.txt

# 冒烟测试只用到提出模型 —— 不需要 mamba-ssm / speechbrain / rotary-embedding 等
```

确认装好：

```bash
PYTHONPATH=src python -c "from tibetan_ss.models import list_models; print(list_models())"
# 预期：['dip_frontend', 'dual_path_mamba', 'identity', 'mossformer2', 'proposed', 'sepreformer', 'tiger']
```

## 3. 配置文件速查（已帮你准备好）

| 文件 | 作用 |
|------|------|
| `configs/data/nict_tib1_speakers.yaml` | 20 位说话人的性别映射（已按 F0 自动填好） |
| `configs/data/nict_tib1.yaml` | 数据预设：指向本地 Tibetan/，DM 开启，噪声关闭，train 离线混合=0 |
| `configs/experiment/smoke_proposed.yaml` | 冒烟实验：提出模型小号版，3 epoch，batch=2，CSV logger |
| `scripts/smoke_test.sh` | 一键串起 prepare → generate val/test → train |

默认路径读取自环境变量；如果你按默认放在 `tibetan_ss/Tibetan/`，可以不设置。

```bash
# 可选：显式指定（或默认都会自动用 repo_root/Tibetan 和 repo_root/data）
export TIBETAN_ROOT="$PWD/Tibetan"
export TIBETAN_SS_OUTPUT="$PWD/data"
export DEMAND_ROOT=""          # 留空 = 噪声禁用
```

## 4. 一条命令：

```bash
scripts/smoke_test.sh
```

这会做 3 件事：

1. **Step 1 — 扫数据并分 split**：
   - 读 20 个说话人目录
   - 按性别 + seed 切：`train 6M+8F / val 1M+2F / test 1M+2F`
   - 生成 `data/manifests/{speakers_{train,val,test}.json, noise_*.json(empty), all_speakers.json}`

2. **Step 2 — 预生成小型 val/test**：
   - train: 跳过（DM 在线合成）
   - val: 200 条混合（overlap mix 分布 80%+20%）
   - test: 200 条混合（overlap mix 分布 70%+30%）
   - 文件落到 `data/mixtures/{val,test}/<mix_id>/{mixture,s1,s2}.wav`

3. **Step 3 — 训练 3 个 epoch**：
   - 提出模型小号（N=128, H=128, TCN 4×1）
   - batch=2, precision=32-true（CPU 可用）
   - CSV logger 写 `outputs/logs/smoke_proposed/`
   - 三阶段 schedule：epoch 0 只 L_main，epoch 1 加 L_rep，epoch 2 开 GAN

## 5. 想分步骤执行

```bash
# Step 1 only
PYTHONPATH=src python -m tibetan_ss.data.scripts.prepare_nict_tib1 \
    --config configs/data/nict_tib1.yaml

# Step 2 only
PYTHONPATH=src python -m tibetan_ss.data.scripts.generate_mixtures \
    --config configs/data/nict_tib1.yaml --splits val test

# Step 3 only
PYTHONPATH=src python -m tibetan_ss.cli.train \
    --config configs/experiment/smoke_proposed.yaml
```

## 6. 跑完后看什么

```bash
# 查看训练 log
cat outputs/logs/smoke_proposed/csv/version_0/metrics.csv | head

# 查看最后 checkpoint
ls outputs/logs/smoke_proposed/checkpoints/

# 用 checkpoint 在 test 上再评估一遍（自定义）
scripts/evaluate.sh configs/experiment/smoke_proposed.yaml \
    outputs/logs/smoke_proposed/checkpoints/last.ckpt
```

冒烟测试成功的标志：

- [ ] `outputs/logs/smoke_proposed/csv/version_0/metrics.csv` 非空，至少有 3 行 epoch 记录
- [ ] `outputs/logs/smoke_proposed/checkpoints/last.ckpt` 存在
- [ ] `val/si_sdr` 第 3 个 epoch 比第 1 个至少**不变差**（分离问题在这种小模型+3 epoch 下通常能跑到 3-6 dB）
- [ ] stdout 三个阶段的 loss 都有数字（`train/loss_main`, `train/loss_rep` 从 epoch 1 开始出现，`train/loss_d` / `train/loss_g` 从 epoch 2 开始出现）

## 7. 关键问答

### Q1 为什么 train.json 是空的？

因为 `configs/data/nict_tib1.yaml` 里 `offline.num_mixtures.train=0`。DM 模式下 train dataset 直接读 `speakers_train.json` 做在线合成，不需要预生成文件。

这意味着：
- 磁盘占用：只 ~2 MB（val+test 400 条混合），而不是几十 GB
- 可以立即开始训练，不用等上 30-60 min 的 offline 生成
- 代价：每次 epoch 的 sample 都不同（不能 bit-for-bit 复现训练轨迹；但 seed 固定下 cacheable）

### Q2 为什么不开噪声？

DEMAND 还没下载。冒烟测试的目的是验证框架流水线，不是复现最终指标。后续想加噪声：

```bash
# 下载 DEMAND 到某处后
export DEMAND_ROOT=/path/to/DEMAND
# 改数据 config
sed -i '' 's/enabled: false/enabled: true/' configs/data/nict_tib1.yaml
# 重跑 prepare + generate
scripts/smoke_test.sh
```

### Q3 性别自动识别是怎么做的？

`configs/data/nict_tib1_speakers.yaml` 里已经填好 20 位说话人的性别（基于 YIN F0 估计：阈值 165 Hz）。如果你人工试听后发现 `049` 实际是 F，直接把那行改成 `"049": F`，重跑 `prepare_nict_tib1.py` 即可。

### Q4 怎么扩展到其它对标模型？

冒烟配置使用了 `engine: gan`（提出模型专用）。其它 5 个模型用 `engine: standard`。跑一个 tiger 小号版冒烟：

```bash
PYTHONPATH=src python -m tibetan_ss.cli.train \
    --config configs/experiment/baseline_tiger.yaml \
    defaults.0.data=nict_tib1 \
    training.trainer.max_epochs=3 \
    training.dataloader.batch_size=2 \
    model.num_blocks=4 \
    training.logger.name=csv
```

（或者照抄 `smoke_proposed.yaml` 做一份 `smoke_tiger.yaml`。）

### Q5 CPU 能跑吗？要多久？

能。在 M3 MacBook 上，3 epoch × 2000 DM samples/epoch × batch=2 大约 **20-30 分钟**。如果你只想最快验证流水线，还可以把 `cache_per_epoch` 从 2000 调到 200：

```bash
scripts/smoke_test.sh \
    data.dynamic_mixing.cache_per_epoch=200
```

这样一个 epoch 几十秒搞定。

### Q6 训到一半 OOM 怎么办？

按顺序尝试：

```bash
# 1. 缩 batch
scripts/smoke_test.sh training.dataloader.batch_size=1

# 2. 缩 segment
scripts/smoke_test.sh \
    training.dataloader.batch_size=1 \
    data.segment.train=2.0

# 3. 换 16-mixed 精度（仅 GPU）
scripts/smoke_test.sh \
    training.dataloader.batch_size=1 \
    training.trainer.precision=16-mixed
```

## 8. 冒烟之后怎么升到 "正式训练"

1. **加 DEMAND 噪声**（见 Q2）
2. **回到完整模型尺寸**：在 experiment config 里删掉 `model:` 块，或直接用 `configs/experiment/proposed.yaml`
3. **增大训练规模**：
   ```yaml
   dynamic_mixing.cache_per_epoch: 20000
   offline.num_mixtures.val:  3000
   offline.num_mixtures.test: 3000
   training.trainer.max_epochs: 200
   training.dataloader.batch_size: 4
   training.early_stop.enabled: true
   ```
4. **打开 TensorBoard 监控**：`training.logger.name: tensorboard`
5. **并行跑 baselines**：`scripts/run_all.sh`（先在每个 experiment 里把 `defaults.0.data=nict_tib1`）
