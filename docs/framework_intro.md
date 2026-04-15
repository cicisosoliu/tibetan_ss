# 框架总览

> 这份文档面向初次接触 `tibetan_ss` 的同学或协作者。读完以后你应该能回答：
> （1）这个框架为什么要这样组织？
> （2）一条训练指令背后发生了什么？
> （3）我要改一个东西时，应该改哪个文件？

## 1. 背景与目标

本仓库承载的是"藏语多人（双人）单通道语音分离"课题的全部实验代码。设计时只有一个核心诉求：

> **5 个对标模型 + 我们自己提出的 Early-Separation 模型，要能在完全相同的数据、相同的训练 loop、相同的评估指标下比较。**

在调研过程中我们发现，直接把每个对标仓库克隆下来各自为战，会带来三个麻烦：

1. **数据管道不一致**：TIGER 用的是 44.1 kHz LibriMix、SepReformer 用 8 kHz WSJ0-2mix、MossFormer2 又有自己的 SpeechBrain recipe，没法直接横向比较
2. **指标口径不一致**：有的算 SI-SNR、有的算 SI-SDR，有的在 `min` 模式（取短的），有的在 `max` 模式
3. **训练 loop 差异**：Dynamic Mixing 的实现、PIT 的实现、batch 调度都不同

所以我们选择了一条"胶水 + 适配器"的路线：**保留每个模型的原始权重/计算图（不重写网络本体）**，在外面裹一个统一的 `BaseSeparator` 接口，然后让数据、训练、评估全部共用一套代码。

## 2. 设计哲学

可以用 4 句话概括：

1. **Everything is a `BaseSeparator`**。所有模型都满足 `forward((B, T) | (B, 1, T)) → (B, 2, T)` 的接口，上层代码不需要知道底下是 TCN、Transformer 还是 Mamba。
2. **Configs are the truth**。任何超参、路径、开关都必须来自 `configs/*.yaml`；代码里零魔术常数。这样做消融实验只需要改 YAML。
3. **Data is immutable once generated**。离线预生成的 `mixture.wav / s1.wav / s2.wav` 跟 manifest 是一一对应的，每次重跑同样 seed 会得到 bit-for-bit 一致的结果。Dynamic Mixing 作为独立开关。
4. **Lightning is the trainer, not the framework**。我们用 Lightning 只是为了省去写 training loop 的样板代码，但核心业务（模型、数据、损失、指标）都是纯 PyTorch，脱离 Lightning 依然能跑。

## 3. 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                       configs/experiment/*.yaml                    │
│      (defaults + overrides -- 单一入口定义一次实验)                │
└──────┬────────────────────┬────────────────┬──────────────────────┘
       │                    │                │
       ▼                    ▼                ▼
┌────────────┐       ┌────────────┐   ┌─────────────────┐
│ data/*.yaml│       │ model/*.yaml│   │ training/*.yaml │
└────┬───────┘       └────┬───────┘   └────────┬────────┘
     │                    │                    │
     ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐
│TibetanMixDataset │  │ build_model(cfg) │  │ SeparationModule   │
│ MixtureSimulator │  │   registry[name] │  │ ProposedGANModule  │
└──────────┬───────┘  └─────────┬────────┘  └──────────┬─────────┘
           │                    │                      │
           └────────────────────┼──────────────────────┘
                                ▼
                    ┌────────────────────────┐
                    │   pl.Trainer.fit/test  │
                    │   TensorBoard / CSV    │
                    │   ModelCheckpoint      │
                    └────────────────────────┘
```

每一层都是单向依赖（config → 数据/模型/训练 → trainer），任意一层都可以独立单测。

## 4. 目录职责矩阵

| 目录 | 干什么 | 什么时候改 |
|------|--------|-----------|
| `configs/data/` | 数据路径、分段、overlap/K/SNR 分布、采样率 | 想换数据集或改采样策略时 |
| `configs/model/` | 每个模型的超参 | 调模型结构 / 换 variant 时 |
| `configs/training/` | Optimizer / Scheduler / batch_size / logger | 调优化策略 / 硬件改变时 |
| `configs/experiment/` | 上面三者的组合 + `tag` + `engine` | 定义一个新实验 |
| `src/tibetan_ss/data/` | `MixtureSimulator` 合成逻辑 + `Dataset` + `DataModule` | 加新数据增强 / 改数据管道 |
| `src/tibetan_ss/models/` | 每个模型一个 adapter 文件 + `registry` 自动注册 | 加新对标模型 |
| `src/tibetan_ss/models/proposed/` | 提出模型的 encoder / branch / decoder / discriminator | 调提出模型结构 |
| `src/tibetan_ss/losses/` | SI-SDR、PIT | 加新损失函数 |
| `src/tibetan_ss/engine/` | 两种 Lightning module + metrics | 改训练 loop / 加指标 |
| `src/tibetan_ss/cli/` | `train.py` / `evaluate.py` / `aggregate_results.py` | 加命令行子命令 |
| `src/tibetan_ss/utils/` | io / logging / seed 管理 | 底层工具改动 |
| `scripts/` | bash 一键脚本 | 改实验流程 / 部署方式 |
| `third_party/` | 对标仓库 clone | 不要手改里面的文件 |
| `tests/` | smoke test | 加/改接口后同步扩展 |

## 5. 一条训练指令的完整生命周期

跑下面这条命令：

```bash
scripts/train.sh configs/experiment/proposed.yaml training.trainer.max_epochs=80
```

后台发生的事：

### Step 1: Config 解析

`cli/train.py::_resolve_defaults` 做一件事：把 experiment YAML 里的 `defaults: [...]` 语义展开成一份扁平 dict。

```yaml
# configs/experiment/proposed.yaml
defaults:
  - /data: sr16k           # 读 configs/data/sr16k.yaml
  - /model: proposed       # 读 configs/model/proposed.yaml
  - /training: default     # 读 configs/training/default.yaml
  - _self_                 # 把本文件的剩余字段合并进来
tag: proposed
engine: gan
seed: 20260415
```

展开后得到的 `cfg` 形如：

```python
{
    "data":     {...},        # 来自 configs/data/sr16k.yaml
    "model":    {...},        # 来自 configs/model/proposed.yaml
    "training": {...},        # 来自 configs/training/default.yaml
    "tag":      "proposed",
    "engine":   "gan",
    "seed":     20260415,
}
```

CLI 再把命令行 override（`training.trainer.max_epochs=80`）合并进去。最后 `cfg` 会原样写到 `outputs/logs/proposed/resolved.yaml`，方便复现。

### Step 2: Seed 固定

```python
set_seed(cfg["seed"])   # 随机种子在 CUDA / Python / NumPy 三层同时固定
```

### Step 3: 数据模块构造

```python
datamodule = TibetanMixDataModule(cfg=cfg["data"], training_cfg=cfg["training"])
# 初始化时没有读任何文件；
# 第一次 trainer.fit() 会触发 datamodule.setup("fit")，此时扫描 manifests 并构造三个 dataset。
```

### Step 4: 模型构造

```python
model = build_model({
    **cfg["model"],
    "sample_rate": cfg["data"]["sample_rate"],
    "num_speakers": cfg["data"]["mixing"]["num_speakers"],
})
# 内部做了：
#   registry[cfg["model"]["name"]](**kwargs) -> BaseSeparator
# 对 proposed 是 ProposedEarlySeparation；对 tiger 是 TIGERAdapter；...
```

### Step 5: Lightning module 选择

```python
if cfg["engine"] == "standard":
    pl_module = SeparationModule(model, cfg["training"], ...)
elif cfg["engine"] == "gan":
    pl_module = ProposedGANModule(model, cfg["training"], ...,
                                  discriminator_cfg=cfg["model"]["discriminator"],
                                  schedule_cfg=cfg["model"]["schedule"])
```

两种 engine 的核心差异：

| | `SeparationModule` | `ProposedGANModule` |
|---|---|---|
| 优化器数 | 1 (AdamW on model) | 2 (AdamW on model + AdamW on discriminator) |
| 自动优化 | 是 | 否（`automatic_optimization = False`） |
| 损失函数 | PIT + SI-SDR | Stage 切换：L_main / +L_rep / +L_D |
| 适用 | 5 个对标 | 提出模型 |

### Step 6: Trainer 初始化与运行

`pl.Trainer` 负责 checkpoint、早停、LR monitor、logger 搭建，然后 `trainer.fit(pl_module, datamodule)` → 训练。训练结束后 `trainer.test(ckpt_path="best")` 自动 load 验证集上 SI-SDRi 最好的那个 checkpoint 做测试。

### Step 7: 评估与落盘

- 指标（SI-SDR / SI-SDRi / PESQ-WB / STOI）每个 epoch 在 val 上计算一次
- checkpoint 默认保留 top-3 + last
- TensorBoard / CSV logs 在 `outputs/logs/<tag>/`

## 6. 核心契约

这是整套框架的"宪法"，任何新代码都必须遵守：

### 6.1 模型契约 — `BaseSeparator`

```python
class BaseSeparator(nn.Module):
    num_speakers: int = 2
    sample_rate:  int = 16000

    def forward(self, mixture: Tensor) -> Tensor:
        # mixture:  (B, T) 或 (B, 1, T)
        # return:   (B, num_speakers, T)
        ...
```

**不能**返回 list/tuple，**不能**返回 (B, T, C)，**不能**比输入长度长/短 > `kernel_size`（适配器需在内部 crop/pad 对齐）。

### 6.2 数据契约 — `TibetanMixDataset.__getitem__`

每个样本返回一个 dict：

```python
{
    "mixture":     Tensor (T,),
    "sources":     Tensor (2, T),
    "meta":        dict   (overlap_ratio, level_diff_db, snr_db, gender_pair, ...),
    "id":          str,
    "sample_rate": int,
}
```

test 的 DataLoader 还会走 `collate_variable_length`，增加一个 `length` Tensor。

### 6.3 Loss 契约 — PIT 一律在外层

每个模型只需要输出 `(B, C, T)`，不要在模型内部做 PIT / 排序。PIT 由 `losses.pit.pit_si_sdr_loss` 统一处理。如果要对齐后的输出做进一步处理（如给判别器），`reorder_sources(est, perm)` 把估计按 PIT 最优排列重新排好。

### 6.4 指标契约 — 只评估对齐后的输出

`evaluate_batch(est, ref, mixture, sr, metric_list)` 要求 `est` 已按 PIT 对齐。SI-SDRi 基准是 `SI-SDR(mixture, ref) → SI-SDR(est, ref)`。

## 7. 配置系统的层叠逻辑

我们的 YAML 不依赖 Hydra 插件，只有 3 个操作：

1. **`defaults`**：展开其它 YAML 到对应的 key
2. **`_self_`**：表示"把当前文件剩余字段合并"
3. **CLI override**：`dot.notation.path=value`，从 `argv` 合进来

效果与 Hydra 一致，但是代码只有 30 行（`cli/train.py::_resolve_defaults`），可 100% 追溯。

### 覆盖顺序（后来者覆盖先来者）

```
data/default.yaml
  ↓ merge
data/sr16k.yaml  (只 override sample_rate=16000)
  ↓ merge
model/<name>.yaml
  ↓ merge
training/default.yaml
  ↓ merge
experiment/<tag>.yaml 本身的 top-level 字段
  ↓ merge
命令行 CLI overrides  ← 最高优先级
```

典型用例：

```bash
# 换采样率、改 batch size、缩减 epoch、减少训练集
scripts/train.sh configs/experiment/proposed.yaml \
    defaults.0.data=sr8k               # 换 8 kHz（直接改 experiment 文件更清晰）
    training.trainer.max_epochs=20 \
    training.dataloader.batch_size=8 \
    data.offline.num_mixtures.train=2000
```

## 8. 三阶段训练的机制

提出模型的三阶段训练（`ProposedGANModule`）不是在 `fit.epoch_end` 上硬切换，而是通过 `current_epoch` 阈值判断：

```python
@property
def _rep_enabled(self): return self.current_epoch >= schedule["rep_from_epoch"]
@property
def _gan_enabled(self): return self.current_epoch >= schedule["gan_from_epoch"]
```

- Stage 1（epoch 0 - 9）：只用 `L_main = PIT(SI-SDR)`
- Stage 2（epoch 10 - 24）：加上 `L_rep = cosine_sim + orth_gram`
- Stage 3（epoch 25+）：再加上判别器的 hinge GAN

这样做的好处：

1. 每个 stage 的边界可以**热更**（只改 config，不改代码）
2. warmup 损失（如 L_rep）可以从 ε 逐渐放大，在 config 里引入 schedule function 即可
3. checkpoint 是一张图：即便中途崩了，resume 也只看 `current_epoch` 自动推断当前 stage

判别器更新顺序（每一个 train step）：

```
1. 前向 → PIT → 对齐 est_aligned
2. 若 gan_enabled:
     D.zero_grad()
     d_real = D(ref), d_fake = D(est_aligned.detach())
     (hinge_D).backward()  → D.step()
3. G.zero_grad()
   total = L_main
   若 rep_enabled: total += lam_rep * L_rep(z_a, z_b)
   若 gan_enabled: total += lam_g * (-D(est_aligned)).mean()
   total.backward()  → G.step()
```

这个顺序保证了判别器不会因为生成器梯度污染而"作弊"，同时 PIT 对齐发生在判别器之前——否则 D 学不到稳定的"单一说话人"概念。

## 9. 依赖矩阵与运行时注意

| 模型 | 核心依赖 | 平台 |
|------|----------|------|
| `identity`       | —                                         | 任意 |
| `proposed`       | torch, torchaudio                         | 任意 |
| `tiger`          | torch, asteroid-filterbanks, rotary-embedding | 任意 |
| `sepreformer`    | torch, yaml                               | 任意 |
| `mossformer2`    | torch, speechbrain                        | 任意 |
| `dual_path_mamba`| torch, speechbrain, mamba-ssm, causal-conv1d | **CUDA only** |
| `dip_frontend`   | torch                                     | 任意 |

**注意事项**

- `mamba-ssm` 需要 CUDA 11.6+ 编译 wheel，Apple Silicon / Intel Mac 上无法跑。所以 `baseline_dual_path_mamba` 实验一定要在 Linux + NVIDIA GPU 机器上启动。
- SpeechBrain 版本在 2024-2026 间有过几次 breaking change；MossFormer2 的 import path 可能变化。`mossformer2.py` 会按列表依次尝试 candidate import，不行就抛一个带清单的 ImportError。
- TIGER 的上游要求 `rotary_embedding_torch` 版本不要太新，如果 `import look2hear.models.tiger` 失败，先 `pip install "rotary_embedding_torch<0.6"`。

## 10. 速查手册

| 我想 | 看/改 |
|------|------|
| 查看已注册模型 | `python -c "from tibetan_ss.models import list_models; print(list_models())"` |
| 跑 smoke test | `pytest tests/ -q` |
| 只跑 3 个 epoch 看看 | `scripts/train.sh <exp.yaml> training.trainer.max_epochs=3` |
| 换采样率 | 编辑 `configs/experiment/<tag>.yaml::defaults`，把 `data: sr16k` 改成 `data: sr8k` |
| 开启动态混合 | `scripts/train.sh ... data.dynamic_mixing.enabled=true` |
| 调 GAN 启动时机 | 改 `configs/model/proposed.yaml::schedule.gan_from_epoch` |
| 导出 best ckpt 做推理 | `scripts/evaluate.sh <exp.yaml> outputs/logs/<tag>/checkpoints/epochNNN.ckpt` |
| 看 TensorBoard | `tensorboard --logdir outputs/logs` |
| 汇总所有实验指标 | `python -m tibetan_ss.cli.aggregate_results --root outputs/logs --output outputs/summary.md` |
