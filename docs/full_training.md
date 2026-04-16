# 从冒烟到正式训练

> 冒烟测试 (`docs/smoke_nict_tib1.md`) 已跑通的前提下，本文告诉你：
> （1）怎么下载 DEMAND 并打开噪声
> （2）怎么把 `smoke_proposed` 升级到**完整的提出模型 + 200 epoch**
> （3）怎么把 5 个对标模型也跑起来做对比

## 目录

1. [下载 DEMAND 噪声](#1-下载-demand-噪声)
2. [打开噪声 + 回到正式训练规模](#2-打开噪声--回到正式训练规模)
3. [跑提出模型的 Formal 实验](#3-跑提出模型的-formal-实验)
4. [跑 5 个对标模型](#4-跑-5-个对标模型)
5. [一键跑完 6 个实验](#5-一键跑完-6-个实验)
6. [监控、断点续训、评估](#6-监控断点续训评估)
7. [迁移到更强硬件](#7-迁移到更强硬件)
8. [文件/配置对照表](#8-文件配置对照表)

---

## 1. 下载 DEMAND 噪声

DEMAND (Diverse Environments Multichannel Acoustic Noise Database) 是语音增强/分离领域的标准噪声库，来源于 Thiemann 等 2013，由 [Zenodo 1227121](https://zenodo.org/records/1227121) 分发。

仓库里已经备好了下载脚本：

```bash
cd "/Users/liuhao/research/speech separation/tibetan_ss"
scripts/download_demand.sh
```

### 脚本做了什么

1. 从 Zenodo 拉取 **17 个场景类别的 16 kHz 压缩包**（SCAFE 上游只提供 48 kHz，默认跳过；想要加上就 `INCLUDE_SCAFE=1 scripts/download_demand.sh`）
2. 解压到 `$REPO_ROOT/DEMAND/<CATEGORY>/ch01.wav ... ch16.wav`
3. 删掉 zip，保留 wav

17 类场景是：
```
DKITCHEN  DLIVING   DWASHING             (Domestic)
NFIELD    NPARK     NRIVER               (Nature)
OHALLWAY  OMEETING  OOFFICE              (Office)
PCAFETER  PRESTO    PSTATION             (Public)
SPSQUARE  STRAFFIC                       (Street)
TBUS      TCAR      TMETRO               (Transport)
```

### 资源占用

| 阶段 | 空间 |
|------|------|
| 下载中 (zip) | ≈ 1.5 GB |
| 解压后 (wav) | ≈ 2.5 GB |
| 下载时长 | 10-30 min（取决于网速） |

### 验证下载结果

```bash
ls DEMAND/
# 应看到 17 个目录：DKITCHEN/ DLIVING/ ... TMETRO/

find DEMAND -name '*.wav' | wc -l
# 应该 272 个 (17 × 16 通道)

python3 -c "
import soundfile as sf
info = sf.info('DEMAND/DKITCHEN/ch01.wav')
print(f'sr={info.samplerate} ch={info.channels} frames={info.frames} dur={info.frames/info.samplerate:.1f}s')
"
# 预期：sr=16000 ch=1 frames=4800000 dur=300.0s
```

### 下载到自定义目录

```bash
scripts/download_demand.sh /data/DEMAND         # 直接传目录
# 或
DEMAND_ROOT=/data/DEMAND scripts/download_demand.sh
```

### 常见下载失败对策

- **中断续传**：直接重跑脚本，`curl --continue-at -` 会续上
- **Zenodo 访问慢**：脚本内置 `--retry 3`；或改用 IPv4：`alias curl='curl -4'`
- **某一类下载失败**：手动重跑单一类别：
  ```bash
  curl -L -o DEMAND/DKITCHEN.zip \
      "https://zenodo.org/records/1227121/files/DKITCHEN_16k.zip?download=1"
  cd DEMAND && unzip DKITCHEN.zip && rm DKITCHEN.zip
  ```

---

## 2. 打开噪声 + 回到正式训练规模

### 2.1 设置环境变量

```bash
# 在 shell 里（或写进 .env / .envrc）
export TIBETAN_ROOT="$PWD/Tibetan"        # 如果默认放在 tibetan_ss/ 根目录下可省略
export DEMAND_ROOT="$PWD/DEMAND"          # 关键：没这个就还是无噪声
export TIBETAN_SS_OUTPUT="$PWD/data"      # 生成数据落盘位置
```

### 2.2 重新跑 prepare + 正式规模混合生成

冒烟测试只生成了 200+200 条 val/test，正式训练要 3k+3k。所以**必须重新跑一次**：

```bash
scripts/prepare_data.sh configs/data/nict_tib1_full.yaml
```

这条命令等价于：

```bash
# 1) 扫说话人（同冒烟）+ 扫 DEMAND 噪声（这次有了）
python -m tibetan_ss.data.scripts.prepare_nict_tib1 \
    --config configs/data/nict_tib1_full.yaml

# 2) 生成正式规模混合（3000 val + 3000 test，train 跳过因为 DM）
python -m tibetan_ss.data.scripts.generate_mixtures \
    --config configs/data/nict_tib1_full.yaml \
    --splits train val test
```

预期输出：

```
[prepare] found 20 speakers in .../Tibetan
   - 006 (F): 400 files
   ...
[prepare] wrote speakers_train.json with 14 speakers
[prepare] wrote speakers_val.json with 3 speakers
[prepare] wrote speakers_test.json with 3 speakers
[prepare] found 272 noise files in .../DEMAND
[prepare] wrote noise_train.json with 217 files
[prepare] wrote noise_val.json with 27 files
[prepare] wrote noise_test.json with 28 files

[mix] skip train (num_mixtures=0; covered by dynamic mixing)
generate val: 100%|████████████| 3000/3000 [00:20<00:00, 145.12it/s]
[mix] wrote data/manifests/val.json with 3000 items
generate test: 100%|████████████| 3000/3000 [00:19<00:00, 155.32it/s]
[mix] wrote data/manifests/test.json with 3000 items
```

生成时间：~1 分钟；磁盘占用：~2 GB（6000 条混合 × 约 ~300 KB/条）。

### 2.3 为什么这次不用等几小时？

因为我们**只对 val/test 做离线生成**，train 完全走 dynamic mixing。相比"全部离线"的 20k+3k+3k 方案（≈ 10-30 min 生成 + 30-50 GB 磁盘），这套方案在正式训练下依然保证了公平对比（val/test 固定），但省掉了 train 预生成的开销。

如果你执意要离线预生成 train，把 `configs/data/nict_tib1_full.yaml` 里的 `offline.num_mixtures.train` 改成 20000，然后 `dynamic_mixing.enabled` 设 false，再重跑 step 2.2 即可（会跑 30-60 min）。

---

## 3. 跑提出模型的 Formal 实验

正式训练 config 已经准备好：

```bash
scripts/train.sh configs/experiment/proposed_formal.yaml
```

### 这份 config 相比 `smoke_proposed.yaml` 的差异

| 项 | smoke | formal |
|---|---|---|
| 模型尺寸 | N=128, H=128, TCN 4×1 | N=512, H=512, TCN 8×3/4/3（`configs/model/proposed.yaml` 原值） |
| epochs | 3 | 200 |
| batch_size | 2 | 4 |
| precision | 32-true | 16-mixed |
| early stopping | 关 | 开（patience=20，monitor `val/si_sdri`） |
| logger | csv | tensorboard |
| data preset | `nict_tib1`（噪声关，val/test 200 条） | `nict_tib1_full`（噪声开，3000 条） |
| dynamic_mixing.cache_per_epoch | 2000 | 20000 |
| schedule (rep/gan_from_epoch) | 1/2（几乎立刻触发） | 10/25（正常 warmup） |

### 跑之前再确认一次

```bash
# 检查数据
ls data/manifests/        # 应有 speakers_*.json, noise_*.json, val.json, test.json
ls data/mixtures/val/ | wc -l   # 3000

# 检查配置会被正确加载（不起 Lightning）
PYTHONPATH=src python -c "
from tibetan_ss.utils.config import load_config
c = load_config('configs/experiment/proposed_formal.yaml')
print('engine=', c['engine'])
print('data.sample_rate=', c['data']['sample_rate'])
print('data.noise.enabled=', c['data']['noise']['enabled'])
print('data.dynamic_mixing.enabled=', c['data']['dynamic_mixing']['enabled'])
print('model.name=', c['model']['name'])
print('model.n_filters=', c['model']['n_filters'])
print('training.trainer.max_epochs=', c['training']['trainer']['max_epochs'])
"
```

预期：
```
engine= gan
data.sample_rate= 16000
data.noise.enabled= True
data.dynamic_mixing.enabled= True
model.name= proposed
model.n_filters= 512
training.trainer.max_epochs= 200
```

### 资源预估

| 硬件 | batch=4 | 每 epoch | 200 epoch |
|------|---------|---------|----------|
| RTX 3090 / 4090 (24 GB) | ✓ | ~5 min | ~15 h |
| RTX 3080 (10 GB) | 需 batch=2 | ~7 min | ~23 h |
| V100 (32 GB) | ✓ | ~6 min | ~18 h |
| A100 (40 GB) | 建议 batch=8 | ~3 min | ~9 h |
| M3 Max CPU | 实际不现实 | ~2-3 h | 停 |

### 如果显存不够

```bash
# 缩 batch
scripts/train.sh configs/experiment/proposed_formal.yaml \
    training.dataloader.batch_size=2

# 或者累积梯度
scripts/train.sh configs/experiment/proposed_formal.yaml \
    training.dataloader.batch_size=2 \
    training.trainer.accumulate_grad_batches=2
```

### 如果你想再缩一下（只为看曲线不追求最终数字）

```bash
scripts/train.sh configs/experiment/proposed_formal.yaml \
    training.trainer.max_epochs=50 \
    data.dynamic_mixing.cache_per_epoch=5000
```

---

## 4. 跑 5 个对标模型

仓库里准备了 5 个以 `_nict` 结尾的对标 experiment：

| 实验 config | 模型 | 额外依赖 |
|------------|------|---------|
| `configs/experiment/baseline_tiger_nict.yaml` | TIGER | `rotary_embedding_torch` `asteroid-filterbanks` |
| `configs/experiment/baseline_sepreformer_nict.yaml` | SepReformer | — |
| `configs/experiment/baseline_dual_path_mamba_nict.yaml` | Dual-path Mamba | CUDA + `mamba-ssm` `causal-conv1d` `speechbrain` |
| `configs/experiment/baseline_mossformer2_nict.yaml` | MossFormer2 | `speechbrain` |
| `configs/experiment/ext_dip_nict.yaml` | DIP Frontend | — |

跑单个：

```bash
scripts/train.sh configs/experiment/baseline_tiger_nict.yaml
```

跑前先装依赖（参考 `docs/reproduce.md` 第 1.2 节）。Dual-path Mamba 只有在 CUDA + Linux 上装得了；Mac 可以跳过。

---

## 5. 一键跑完 6 个实验

```bash
scripts/run_all.sh nict
```

（默认就是 `nict` 模式；传 `public` 会跑原本基于 WSJ0/LibriMix 的配置，你现在不用管。）

这会：

1. 顺序训练 proposed_formal + 5 个 baseline_*_nict
2. 训完后调用 `aggregate_results.py` 把所有实验的 test 指标拉到 `outputs/summary.md`

在单张 A100 上约 **2-3 天**跑完；3090 约 4 天。

### 想选跑部分

直接编辑 `scripts/run_all.sh` 的 `EXPERIMENTS=()` 列表，或者手动展开：

```bash
for exp in proposed_formal baseline_tiger_nict baseline_sepreformer_nict; do
    scripts/train.sh configs/experiment/${exp}.yaml
done

python -m tibetan_ss.cli.aggregate_results \
    --root outputs/logs --output outputs/summary.md
```

---

## 6. 监控、断点续训、评估

### 6.1 TensorBoard 监控

```bash
tensorboard --logdir outputs/logs --port 6006
```

关键曲线：

- `train/loss_main` — 正常应该平滑下降
- `train/loss_rep`  — epoch 10 左右从 0 开始出现并趋近某个 plateau
- `train/loss_d` / `train/loss_g` — epoch 25 左右开始，两者应围绕各自目标值震荡
- `val/si_sdri`    — 核心指标，这条曲线爬升到稳定就说明训练正常

### 6.2 断点续训

如果训练中断了（机器重启、OOM、手动 Ctrl-C），接着跑：

```bash
scripts/train.sh configs/experiment/proposed_formal.yaml \
    +trainer.ckpt_path=outputs/logs/proposed_formal/checkpoints/last.ckpt
```

> CLI `+` 前缀告诉 Hydra 这是新增字段。实际上这个字段会透传到 Lightning 的 `trainer.fit(ckpt_path=...)`。

### 6.3 提前结束拿 best ckpt 做 test

训练 loop 跑到 `max_epochs` 或早停后，自动在 best ckpt 上做 test（代码里默认 `trainer.test(ckpt_path="best")`）。如果你想用任意 ckpt 重新评估：

```bash
scripts/evaluate.sh configs/experiment/proposed_formal.yaml \
    outputs/logs/proposed_formal/checkpoints/epoch120-sisdri14.230.ckpt \
    --output outputs/logs/proposed_formal/manual_test.json
```

---

## 7. 迁移到更强硬件

### 7.1 本地多卡

```bash
scripts/train.sh configs/experiment/proposed_formal.yaml \
    training.trainer.devices=4 \
    training.trainer.strategy=ddp
```

batch 会被 DDP 自动切分到每张卡（每卡 batch=4 → 总 batch=16）。LR 最好线性放大：

```bash
scripts/train.sh configs/experiment/proposed_formal.yaml \
    training.trainer.devices=4 \
    training.trainer.strategy=ddp \
    training.optimizer.lr=4.0e-3
```

### 7.2 云 GPU（按小时计费）

- Vast.ai / RunPod / Lambda Cloud：把整个 `tibetan_ss/` + `data/` + `DEMAND/` + `Tibetan/` 打包上传；在远端 `pip install -e .` 然后 `scripts/run_all.sh`。
- Modal（serverless）：参考仓库里的 `serverless-modal` skill，需要手写 Modal 入口把 dataset 挂到 Volume。

### 7.3 混合精度 / BF16

A100/H100 上建议 BF16（比 FP16 更稳）：

```bash
scripts/train.sh configs/experiment/proposed_formal.yaml \
    training.trainer.precision=bf16-mixed
```

---

## 8. 文件/配置对照表

### 新增文件

| 文件 | 作用 |
|------|------|
| `scripts/download_demand.sh` | DEMAND 一键下载 |
| `configs/data/nict_tib1_full.yaml` | 正式训练数据预设（噪声开、3k val/test、DM） |
| `configs/experiment/proposed_formal.yaml` | 提出模型正式 200 epoch |
| `configs/experiment/baseline_tiger_nict.yaml` 等 5 个 | 5 个对标模型的 NICT-Tib1 formal 配置 |
| `docs/full_training.md` | 本文 |

### Experiment config 对照

| 场景 | 提出模型 | baselines |
|------|---------|-----------|
| 冒烟（CPU 可跑） | `smoke_proposed.yaml` | 自己写 `smoke_<name>.yaml` 或 CLI 覆盖 |
| 正式（NICT-Tib1 + DEMAND） | `proposed_formal.yaml` | `baseline_<name>_nict.yaml` |
| 公共数据集（WSJ0 / LibriMix 等原论文设定） | `proposed.yaml` | `baseline_<name>.yaml` |

### Data config 对照

| 预设 | 噪声 | val/test 数 | DM | 适用 |
|------|------|-----------|-----|------|
| `nict_tib1.yaml` | off | 200 | on | 冒烟 / 脱机调试 |
| `nict_tib1_full.yaml` | on | 3000 | on | 正式训练（本文） |
| `sr16k.yaml` / `sr8k.yaml` | 取决于 default | 取决于 default | off | 公共数据集的纯采样率切换 |

---

## 9. 常见问题

### 9.1 跑 formal 时发现 val SI-SDR 几个 epoch 都是负数

正常。提出模型的前几个 epoch 只在跑 L_main，双分支还没分化；等 epoch 10 加 L_rep、epoch 25 加 GAN 之后 SI-SDRi 会继续爬升。如果到 epoch 30 还在原地踏步，检查：

- 学习率是不是太大（默认 1e-3 已是上限；如果 batch 扩大 4×，应该调 2-4e-3）
- GAN 启动时 L_d 是不是爆炸（看 `train/loss_d` 曲线，>5 就太猛，把 `model.schedule.gan_weight` 从 0.1 降到 0.05）

### 9.2 DEMAND 的 16 个 channel 是重复的吗？

不是。同一场景的 16 个通道来自一个 16-mic 阵列，空间位置不同，录到的背景声稍有差异。对我们的单通道训练来说，相当于把每个场景的噪声量扩大 16 倍，是 free 的数据增强。

### 9.3 我想在训练时看某条验证样本的音频

在 TensorBoard 里 Lightning 默认没把 audio 写进去。要加的话在 `ProposedGANModule.validation_step` 结尾调用 `self.logger.experiment.add_audio(...)`。但跑正式实验前改这段不是必要的，导出 best ckpt 之后用 `scripts/evaluate.sh` + 自己写个播放脚本更干净。

### 9.4 可以只跑 3 个对标而不是 5 个吗？

可以。`scripts/run_all.sh` 开头的 `EXPERIMENTS=()` 直接留你关心的几个即可。
