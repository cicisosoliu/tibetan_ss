# 复现步骤

> 预估时间：单卡 A100/RTX4090 上，6 个实验总计 ~72 h（每个 10-15 h 训练到早停）。
> 若只是 smoke-test，把 `training.trainer.max_epochs=3` 即可在 1 h 内过一遍 6 个模型。

## 0. 硬件 / 环境

- OS：macOS / Linux；Windows 未测
- Python：3.10+
- GPU：任意 NVIDIA（Mamba 需 CUDA）；CPU 也能跑 TIGER/SepReformer/Proposed/DIP 的 smoke-test
- 磁盘：预留 ≥ 80 GB 给 `TIBETAN_SS_OUTPUT`（30k 混合 × 3 split × ~5 s × 16kHz × 3 wav）

## 1. 一次性准备

```bash
# 1.1 安装
conda create -n tibetan_ss python=3.10 -y
conda activate tibetan_ss
cd tibetan_ss
pip install -e .
pip install -r requirements.txt

# 1.2 按需装对标依赖
pip install speechbrain                          # MossFormer2 + Dual-path Mamba
pip install rotary_embedding_torch asteroid-filterbanks  # TIGER
# CUDA 环境：
pip install mamba-ssm causal-conv1d

# 1.3 克隆对标仓库
cd third_party
git clone --depth 1 https://github.com/xi-j/Mamba-TasNet.git
git clone --depth 1 https://github.com/JusperLee/TIGER.git
git clone --depth 1 https://github.com/dmlguq456/SepReformer.git
cd ..

# 1.4 指定数据路径
cat > .env <<'EOF'
TIBETAN_ROOT=/data/NICT-Tib1
DEMAND_ROOT=/data/DEMAND
TIBETAN_SS_OUTPUT=/data/tibetan_ss_output
EOF
set -a; source .env; set +a
```

## 2. 数据生成

```bash
scripts/prepare_data.sh configs/data/sr16k.yaml
```

运行完成后检查 `${TIBETAN_SS_OUTPUT}/manifests/` 下应当有：

```
all_speakers.json
noise_{train,val,test}.json
speakers_{train,val,test}.json
train.json   val.json   test.json
```

音频放在 `${TIBETAN_SS_OUTPUT}/mixtures/{train,val,test}/<mix_id>/...`。

若 NICT-Tib1 目录命名不匹配默认正则，加 `--speaker-regex` 参数重跑：

```bash
python -m tibetan_ss.data.scripts.prepare_nict_tib1 \
    --config configs/data/sr16k.yaml \
    --speaker-regex '^(?P<gender>[MFＭＦ])(?P<sid>\d+)'
```

## 3. 训练单个实验

```bash
scripts/train.sh configs/experiment/proposed.yaml
```

日志默认写到 `outputs/logs/proposed/`，checkpoint 在 `outputs/logs/proposed/checkpoints/`，TensorBoard 在 `outputs/logs/proposed/tb/`。

Hydra-style 覆盖任意字段：

```bash
scripts/train.sh configs/experiment/baseline_tiger.yaml \
    training.trainer.max_epochs=5 \
    training.dataloader.batch_size=8 \
    data.mixing.overlap.train.high=0.5
```

## 4. 评估

```bash
scripts/evaluate.sh configs/experiment/proposed.yaml \
    outputs/logs/proposed/checkpoints/last.ckpt \
    --output outputs/logs/proposed/test_metrics.json
```

## 5. 一键对齐所有模型

```bash
scripts/run_all.sh
```

该脚本会顺序训练 6 个 experiment，然后调用 `aggregate_results.py` 把每个实验的最终 test 指标拉到一张 Markdown 表 `outputs/summary.md`。

## 6. 调试 / smoke

```bash
# 运行 7 个 smoke test（不需要 GPU 或重型依赖）
pytest tests/ -q

# 快跑 3 个 epoch
scripts/train.sh configs/experiment/proposed.yaml \
    training.trainer.max_epochs=3 \
    data.offline.num_mixtures.train=200 \
    data.offline.num_mixtures.val=40 \
    data.offline.num_mixtures.test=40
```

## 7. 常见问题

| 症状 | 可能原因 | 处理 |
|------|----------|------|
| `ModuleNotFoundError: mamba_ssm` | CUDA + mamba-ssm 未装 | `pip install mamba-ssm causal-conv1d` |
| `Could not locate MossFormer2` | SpeechBrain 版本太旧/太新 | 升级：`pip install -U speechbrain`；仍不行请看 `_CANDIDATE_PATHS` 并补齐 |
| `FileNotFoundError: third_party/TIGER` | 没 clone | 按第 1.3 步 clone |
| PESQ 全部是 NaN | wav 太短（< 0.5 s）或采样率不是 8k/16k | PESQ 要求的长度 & 采样率；检查 `segment.test` 与 `sample_rate` |
| 训练 loss 到 Stage 3 突然发散 | GAN 过早介入 | 增大 `schedule.gan_from_epoch`、或把 `gan_weight` 调小到 0.05 |
