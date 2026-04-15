# 新增对标模型教程

> 本篇是个完整的 walkthrough：从在 arXiv 上看到一篇新论文，到把它接入 `tibetan_ss` 框架并跑出第一份指标，一共要做哪些事、有哪些坑。

## 目录

1. [集成前的三个判断](#1-集成前的三个判断)
2. [三种集成路径](#2-三种集成路径)
3. [完整模板：路径 A（clone + wrap）](#3-完整模板路径-aclone--wrap)
4. [完整模板：路径 B（复现）](#4-完整模板路径-b复现)
5. [完整模板：路径 C（调 SpeechBrain / Asteroid）](#5-完整模板路径-c调-speechbrain--asteroid)
6. [注册 + 跑通的最小闭环](#6-注册--跑通的最小闭环)
7. [Checklist（每次新增都过一遍）](#7-checklist每次新增都过一遍)
8. [常见坑](#8-常见坑)
9. [端到端案例：集成"FooSep 2026"](#9-端到端案例集成foosep-2026)

---

## 1. 集成前的三个判断

拿到一篇新 paper，先回答三个问题：

### 1.1 输入/输出与我们的契约一致吗？

我们要求：`forward((B, T)) -> (B, 2, T)`。现实中上游模型可能是：

| 上游输出 | 怎么办 |
|----------|--------|
| `(B, 2, T)` | 直接用 |
| `(B, T, 2)` | `.transpose(1, 2)` |
| `list[Tensor(B, T)]` of length 2 | `torch.stack(..., dim=1)` |
| `(audio, audio_aux)` 元组（SepReformer） | 丢 aux，只取 audio，再 stack |
| `{"est_sources": ..., ...}` dict | 取对应 key |
| 可变长度（比 input 长或短） | 在 adapter 里 crop/pad 对齐 |

### 1.2 这个模型的前置依赖有多重？

- 只依赖 torch：**路径 A（clone+wrap）或路径 B（复现）**
- 依赖 speechbrain / asteroid：**路径 C**
- 依赖需要 CUDA 编译的 kernel（mamba-ssm、flash-attn）：还是路径 A，但在 `configs/experiment/<tag>.yaml` 里注明 `requires_cuda: true`，并在 README 增加依赖说明
- 依赖某个没维护的、上游代码都跑不起来的包：**考虑放弃**，或只复现核心 block（路径 B）

### 1.3 论文有没有开放训练权重？

- 有：快速通过复现权重跑推理验证"是否接入正确"
- 无：只好在 Tibetan 上从零训练；那就把 smoke test 写扎实点，免得训练到一半才发现 bug

---

## 2. 三种集成路径

| 路径 | 适用场景 | 工作量 | 维护难度 |
|------|---------|--------|---------|
| A. clone + wrap | 上游代码清爽、PyTorch 原生、forward 签名清晰 | 低（~50 LOC adapter） | 低（仅锁 git commit） |
| B. 从论文复现 | 上游代码不公开 / 代码一坨 / 想完全掌控细节 | 高（~300-1000 LOC） | 中（需要单测） |
| C. 调现成框架 | 该模型已经被 SpeechBrain / Asteroid 实现 | 极低 | 中（跟上游版本漂移敏感） |

我们仓库里已有的例子：

- **路径 A**：`tiger.py`（from look2hear...）、`sepreformer.py`（from SepReformer.models...）
- **路径 C**：`mossformer2.py`（from speechbrain.lobes.models...）、`dual_path_mamba.py`（SpeechBrain Encoder/Decoder + Mamba-TasNet 的 MambaBlocks）
- **路径 B**：`dip_frontend.py`（只有数据集代码，按论文描述复现）

---

## 3. 完整模板：路径 A（clone + wrap）

### Step 1 — Clone 到 `third_party/`

```bash
cd tibetan_ss/third_party
git clone --depth 1 https://github.com/ORG/REPO.git
cd REPO && git log -1 > ../REPO_commit.txt   # 记录 commit，方便未来追踪
```

### Step 2 — 写 adapter 骨架

`src/tibetan_ss/models/<new_name>.py`：

```python
"""Adapter for <Model Name> (Author et al., Venue Year).

Upstream: https://github.com/ORG/REPO
"""

from __future__ import annotations

from typing import Any

import torch

from ._thirdparty_path import register_thirdparty
from .base import BaseSeparator
from .registry import register


class NewModelAdapter(BaseSeparator):
    """Wraps <UpstreamClass> as a BaseSeparator."""

    def __init__(self, sample_rate: int = 16000, num_speakers: int = 2, **kwargs: Any):
        super().__init__(num_speakers=num_speakers, sample_rate=sample_rate)
        register_thirdparty("REPO")                      # 把 third_party/REPO 挂到 sys.path
        from some.module.path import UpstreamClass       # 延后 import，免得 registry 一次性爆
        hparams = {
            "hparam_a": kwargs.get("hparam_a", 128),
            "hparam_b": kwargs.get("hparam_b", 4),
            "num_sources": num_speakers,
            "sample_rate": sample_rate,
        }
        self.model = UpstreamClass(**hparams)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        mix = self._prepare_input(mixture)               # (B, T)
        out = self.model(mix)                             # 上游输出
        # --- 按需把输出归一成 (B, C, T) ---
        if isinstance(out, (list, tuple)):
            out = out[0]                                  # 丢掉 aux
        if out.ndim == 3 and out.shape[-1] == self.num_speakers:
            out = out.transpose(1, 2).contiguous()        # (B, T, C) → (B, C, T)
        # --- 对齐长度 ---
        T = mix.shape[-1]
        if out.shape[-1] > T:
            out = out[..., :T]
        elif out.shape[-1] < T:
            out = torch.nn.functional.pad(out, (0, T - out.shape[-1]))
        return out


@register("new_model")
def build_new_model(sample_rate: int = 16000, num_speakers: int = 2, **kwargs):
    return NewModelAdapter(sample_rate=sample_rate, num_speakers=num_speakers, **kwargs)
```

### Step 3 — 让 registry 自动发现

编辑 `src/tibetan_ss/models/registry.py`，在 `_safe_import` 列表里补一行：

```python
_safe_import("tibetan_ss.models.new_model")
```

`_safe_import` 的作用是：如果 new_model 的依赖还没装，registry 会 print 一个 skip 消息但不会崩 —— 这样没装该依赖的用户依然能用其它模型。

### Step 4 — 写 model config

`configs/model/new_model.yaml`：

```yaml
name: new_model
hparam_a: 128
hparam_b: 4
# 按需补齐其它可调字段
```

### Step 5 — 写 experiment config

`configs/experiment/baseline_new_model.yaml`：

```yaml
defaults:
  - /data:     sr16k
  - /model:    new_model
  - /training: default
  - _self_
tag: baseline_new_model
engine: standard                         # 除非你的模型也要 GAN 训练
seed: 20260415
```

### Step 6 — 加测试

在 `tests/test_smoke.py` 最下面加：

```python
def test_new_model_forward() -> None:
    from tibetan_ss.models import build_model
    model = build_model({"name": "new_model", "sample_rate": 16000, "num_speakers": 2})
    x = torch.randn(2, 16000)
    y = model(x)
    assert y.shape == (2, 2, 16000)
```

### Step 7 — 跑 smoke

```bash
pytest tests/test_smoke.py::test_new_model_forward -q
scripts/train.sh configs/experiment/baseline_new_model.yaml \
    training.trainer.max_epochs=1 \
    data.offline.num_mixtures.train=100 \
    data.offline.num_mixtures.val=20 \
    data.offline.num_mixtures.test=20
```

只要第一个 epoch 能走完并落 checkpoint，就基本接通了，下一步是大规模训练。

---

## 4. 完整模板：路径 B（复现）

如果要从论文复现，建议按"**论文 section 顺序建子模块**"走：

### 4.1 建子包

```
src/tibetan_ss/models/new_model/
├── __init__.py
├── blocks.py        # 论文里的基础 block
├── encoder.py       # § 3.1
├── separator.py     # § 3.2
├── decoder.py       # § 3.3
└── model.py         # 组合 + @register
```

### 4.2 每个 block 先单测

```python
# tests/test_new_model_blocks.py
from tibetan_ss.models.new_model.blocks import FooBlock
import torch

def test_foo_block_shape():
    b = FooBlock(dim=64, kernel=3)
    x = torch.randn(2, 64, 100)
    assert b(x).shape == (2, 64, 100)
```

### 4.3 整合后与论文的 spec 对数

- 参数量：`sum(p.numel() for p in model.parameters())` 跟论文 Table X 对
- MACs / FLOPs：`from fvcore.nn import FlopCountAnalysis`
- 推理时延：`torch.cuda.synchronize(); time.perf_counter()` 取 100 次均值

对不上就回到 section 级别一个个 block 检查。

### 4.4 在 `configs/model/new_model.yaml` 里把**所有**超参暴露出来

不要在代码里藏魔术常数。理想状态是：只读 YAML 就能把这个模型复现出论文报的数。

---

## 5. 完整模板：路径 C（调 SpeechBrain / Asteroid）

以 Asteroid 为例（大多数经典模型已实现）：

```python
"""Adapter for ConvTasNet (Luo & Mesgarani 2019) via Asteroid."""

from __future__ import annotations

import torch

from .base import BaseSeparator
from .registry import register


class ConvTasNetAsteroidAdapter(BaseSeparator):
    def __init__(self, sample_rate: int = 16000, num_speakers: int = 2, **kwargs):
        super().__init__(num_speakers=num_speakers, sample_rate=sample_rate)
        from asteroid.models import ConvTasNet               # pip install asteroid
        self.model = ConvTasNet(
            n_src=num_speakers,
            sample_rate=sample_rate,
            **kwargs,
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        mix = self._prepare_input(mixture)
        return self.model(mix)                                # Asteroid 返回 (B, C, T)


@register("conv_tasnet")
def build_conv_tasnet(sample_rate: int = 16000, num_speakers: int = 2, **kwargs):
    return ConvTasNetAsteroidAdapter(sample_rate=sample_rate, num_speakers=num_speakers, **kwargs)
```

SpeechBrain 类似，只是 import path 是 `speechbrain.lobes.models.xxx`。

---

## 6. 注册 + 跑通的最小闭环

把上面 Step 1-7 总结成"每次新增模型都要做的 7 件事"：

```
1. clone 仓库到 third_party/<REPO>/              （路径 A）
2. 写 src/tibetan_ss/models/<name>.py            （adapter + @register）
3. 编辑 registry.py, 加 _safe_import("tibetan_ss.models.<name>")
4. 写 configs/model/<name>.yaml
5. 写 configs/experiment/baseline_<name>.yaml
6. 在 tests/test_smoke.py 里加 test_<name>_forward
7. pytest 通过 → scripts/train.sh ... max_epochs=1 → 通过 → 上真机跑
```

7 步都通过的话，你的新模型就与整个 pipeline 完全对齐了。

---

## 7. Checklist（每次新增都过一遍）

### 7.1 代码契约

- [ ] 继承了 `BaseSeparator`，没有直接 `nn.Module`
- [ ] `forward(mixture)` 接收 `(B, T)` 或 `(B, 1, T)` — 内部 `self._prepare_input(mixture)`
- [ ] 返回 `(B, num_speakers, T)`，且 T 等于输入 T（crop/pad 对齐）
- [ ] 没有在 `forward` 内部做 PIT 或排序
- [ ] 没有在模型内部直接 `.cuda()` — 让 Lightning 管

### 7.2 Registry

- [ ] 用了 `@register("<unique_name>")`，且名字不与已存在的冲突
- [ ] `registry.py` 里加了对应的 `_safe_import`
- [ ] 缺依赖时 `_safe_import` 输出清晰的 skip 消息

### 7.3 Config

- [ ] `configs/model/<name>.yaml` 顶上有 `name: <unique_name>`
- [ ] 所有可调超参都暴露为 YAML 字段（没有魔术常数）
- [ ] `configs/experiment/baseline_<name>.yaml` 用 `defaults` 引用 data/model/training
- [ ] 必要时覆写 `training.dataloader.batch_size`（如果模型特别吃显存）

### 7.4 数据兼容

- [ ] 模型的 `kernel_size` / `win` / `stride` 与目标采样率匹配
- [ ] 如果模型要求 8 kHz，experiment config 用 `sr8k`
- [ ] 如果模型要求固定长度（比如 32000 samples），在 `data.segment.train` 覆写

### 7.5 测试

- [ ] `tests/test_smoke.py` 里加了最小前向测试（形状 + finite）
- [ ] 本地可以跑 `pytest tests/ -q` 全绿
- [ ] `scripts/train.sh ... max_epochs=1` 能跑完一个 epoch 不崩

### 7.6 文档

- [ ] `docs/models.md` 里新开一节介绍这个模型（出处 + 依赖 + 关键实现点）
- [ ] `README.md` 的模型清单表里加一行
- [ ] `third_party/README.md` 里加 clone 指令（如果走路径 A）

---

## 8. 常见坑

### 8.1 "上游输出 (B, num_spks, T)，但我跑起来形状是 (B, T, num_spks)"

上游 forward 返回的 shape 不是文档里说的那个。**写一个单测打印 shape**：

```python
>>> model = NewModelAdapter(sample_rate=16000)
>>> out = model.model(torch.randn(2, 16000))
>>> print(out.shape)
```

如果是 `(2, 16000, 2)`，就在 adapter 里 `.transpose(1, 2)`。

### 8.2 "训练到一半 SI-SDR 完全不增长"

90% 的情况是 PIT 没生效 —— 模型内部已经做了排序，外层 PIT 又做了一次，导致梯度被拧成乱码。检查：

```python
# 期望：模型对 (mix) 的不同调用顺序无关
out1 = model(mix)
out2 = model(mix)
assert torch.allclose(out1, out2)      # 通过（不要有 random sort 进 forward）
```

### 8.3 "MossFormer2 forward 抛 CUDA OOM，但 batch=1 了"

SpeechBrain 里的 MossFormer2 默认 segment 30 s。让它听我们的 3 s 输入：在 adapter 里把内部的 `chunk_size` 调小，或覆写 `configs/model/mossformer2.yaml::chunk_size`。

### 8.4 "SepReformer 第二次训练就报 `logger_wraps` already defined"

SepReformer 上游用 `sys.path.append('../')` + 全局 `utils.decorators` 等 anti-pattern。adapter 里的 `register_thirdparty` 只 insert 一次到 `sys.path[0]`，但多次 import 会重复执行 `logger_wraps` 装饰器。处理方法：在 `sepreformer.py` 里 catch `ValueError` 或用 `importlib.invalidate_caches()`。

### 8.5 "我的模型训练时慢得要命，推理却很快"

检查 `train() / eval()` 状态切换。SepReformer / DIP 的 `BatchNorm / Dropout` 在 eval 下才有确定行为，但上游代码有时忘了分开处理：

```python
class NewModelAdapter(BaseSeparator):
    def train(self, mode=True):
        super().train(mode)
        # 如果上游模型有自己独立的 train flag 要同步
        if hasattr(self.model, "training_mode"):
            self.model.training_mode = mode
        return self
```

### 8.6 "上游模型需要 spectrogram 输入但我们给的是 waveform"

在 adapter 的 `forward` 里加一道 STFT：

```python
def forward(self, mixture):
    mix = self._prepare_input(mixture)
    spec = torch.stft(mix, n_fft=512, hop_length=128,
                      window=torch.hann_window(512, device=mix.device),
                      return_complex=True)
    # mix → spec → 模型 → 复数 mask → 乘回 spec → istft → est
    ...
```

### 8.7 "模型推理时 requires_grad 依然开着"

`evaluate.py` 里 Lightning 已经调 `trainer.test(..., model.eval())`，但如果你在 adapter 里引入了 `self.some_buffer.detach()` 或自定义 `no_grad` 上下文，可能会和 Lightning 冲突。原则：**adapter 不要自己管 `torch.no_grad`**，让外层来做。

### 8.8 "SpeechBrain 更新后 import path 变了"

在 `mossformer2.py::_CANDIDATE_PATHS` 里加新的候选路径；最底部保留最老的路径作为 fallback。如果是自己新接入的 SpeechBrain 模型，最好写 3-5 个 candidate 来兜底。

---

## 9. 端到端案例：集成"FooSep 2026"

假设我们要集成 ICASSP 2026 的 "FooSep: Foo-based Speaker Separation"：

### 论文与代码速览
- GitHub: `https://github.com/foo-lab/FooSep`
- 要求输入：`(B, 1, T)` 的 16 kHz waveform
- 输出：`(B, num_spks, T)`
- 依赖：`torch`, `einops`
- 核心超参：`n_channels=128, n_blocks=8, downsample_factor=4`

### Step 1 — Clone

```bash
cd tibetan_ss/third_party
git clone --depth 1 https://github.com/foo-lab/FooSep.git
cd FooSep && ls
# expected: foosep/ __init__.py model.py blocks.py README.md
```

### Step 2 — 本地试跑一下原生 forward

```python
import sys; sys.path.insert(0, "third_party/FooSep")
from foosep.model import FooSep
import torch
m = FooSep(n_channels=128, n_blocks=8, num_spks=2)
print(m(torch.randn(2, 1, 16000)).shape)     # 期望 (2, 2, 16000)
```

这一步确认了"签名、输出形状、输入 shape 要求"三件事。

### Step 3 — 写 adapter

`src/tibetan_ss/models/foosep.py`：

```python
"""Adapter for FooSep (Foo et al., ICASSP 2026).

Upstream: https://github.com/foo-lab/FooSep
"""

from __future__ import annotations

from typing import Any

import torch

from ._thirdparty_path import register_thirdparty
from .base import BaseSeparator
from .registry import register


class FooSepAdapter(BaseSeparator):
    def __init__(self,
                 sample_rate: int = 16000, num_speakers: int = 2,
                 n_channels: int = 128, n_blocks: int = 8,
                 downsample_factor: int = 4,
                 **_: Any):
        super().__init__(num_speakers=num_speakers, sample_rate=sample_rate)
        register_thirdparty("FooSep")
        from foosep.model import FooSep
        self.model = FooSep(
            n_channels=n_channels,
            n_blocks=n_blocks,
            downsample_factor=downsample_factor,
            num_spks=num_speakers,
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        mix = self._prepare_input(mixture)               # (B, T)
        mix = mix.unsqueeze(1)                            # (B, 1, T) 因为 FooSep 要这个
        est = self.model(mix)                             # (B, num_spks, T)
        T = mixture.shape[-1]
        if est.shape[-1] > T:
            est = est[..., :T]
        elif est.shape[-1] < T:
            est = torch.nn.functional.pad(est, (0, T - est.shape[-1]))
        return est


@register("foosep")
def build_foosep(sample_rate: int = 16000, num_speakers: int = 2, **kwargs) -> FooSepAdapter:
    return FooSepAdapter(sample_rate=sample_rate, num_speakers=num_speakers, **kwargs)
```

### Step 4 — 注册自动导入

`src/tibetan_ss/models/registry.py` 底部加：

```python
_safe_import("tibetan_ss.models.foosep")
```

### Step 5 — Config

`configs/model/foosep.yaml`：

```yaml
name: foosep
n_channels: 128
n_blocks: 8
downsample_factor: 4
```

`configs/experiment/baseline_foosep.yaml`：

```yaml
defaults:
  - /data:     sr16k
  - /model:    foosep
  - /training: default
  - _self_
tag: baseline_foosep
engine: standard
seed: 20260415
```

### Step 6 — Test

`tests/test_smoke.py` 加：

```python
def test_foosep_forward() -> None:
    from tibetan_ss.models import build_model
    model = build_model({"name": "foosep", "sample_rate": 16000, "num_speakers": 2,
                         "n_channels": 32, "n_blocks": 2})     # 小超参跑快
    y = model(torch.randn(2, 16000))
    assert y.shape == (2, 2, 16000)
```

### Step 7 — 跑 smoke

```bash
pytest tests/test_smoke.py::test_foosep_forward -q

scripts/train.sh configs/experiment/baseline_foosep.yaml \
    training.trainer.max_epochs=1 \
    data.offline.num_mixtures.train=200 \
    data.offline.num_mixtures.val=40 \
    data.offline.num_mixtures.test=40 \
    training.dataloader.batch_size=2
```

1 个 epoch 能跑完并产出 `outputs/logs/baseline_foosep/checkpoints/last.ckpt`？✅ 接入完成，接下来上真机跑 200 epoch 即可。

### Step 8 — 写文档

在 `docs/models.md` 开一节：

```markdown
## FooSep

- 论文：Foo et al., ICASSP 2026
- 上游：https://github.com/foo-lab/FooSep
- 适配器：src/tibetan_ss/models/foosep.py::FooSepAdapter
- 依赖：einops
- 实现要点：上游输入是 (B,1,T)，adapter 内 squeeze；默认超参对齐论文 Table 2 Medium 配置。
```

在 `README.md` 的模型清单表里加一行。

大功告成。

---

## 附录：`BaseSeparator` 完整 spec

```python
from tibetan_ss.models.base import BaseSeparator
import torch

help(BaseSeparator)
# class BaseSeparator(torch.nn.Module):
#     num_speakers: int = 2
#     sample_rate:  int = 16000
#
#     def __init__(self, num_speakers=2, sample_rate=16000)
#
#     def _prepare_input(self, mixture: Tensor) -> Tensor:
#         """接收 (B, T) 或 (B, 1, T)；返回 (B, T)。"""
#
#     def forward(self, mixture: Tensor) -> Tensor:
#         """子类必须实现；返回 (B, num_speakers, T)。"""
```

记住：**你写的 adapter 最终要让下面这条语句 pass**：

```python
model = build_model({"name": "<你的名字>", "sample_rate": 16000, "num_speakers": 2})
x = torch.randn(2, 16000)
y = model(x)
assert y.shape == (2, 2, 16000)
```

能过这条的模型，就自动吃我们的数据 / 训练 / 评估 / 聚合所有基础设施；不能过，就回去看 Checklist。
