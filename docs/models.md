# 模型说明

所有模型共享统一接口：

```python
from tibetan_ss.models import build_model
model = build_model({"name": "<model_name>", "sample_rate": 16000,
                     "num_speakers": 2, **extra_kwargs})
est = model(mixture)      # mixture: (B, T)   → est: (B, 2, T)
```

关键契约：

- 输入 `(B, T)` 或 `(B, 1, T)` — 内部自动归一
- 输出 `(B, num_speakers, T)`；若底层模型输出长度略有偏差，适配器会做 crop/pad 对齐
- 提出模型额外支持 `forward(mix, return_aux=True)`，返回 `{z_a, z_b, encoded, features}` 供 GAN 训练使用

---

## Dual-path Mamba

- 论文：Jiang et al., *Dual-path Mamba: Short and Long-term Bidirectional Selective Structured State Space Models for Speech Separation*, ICASSP 2025
- 上游：[xi-j/Mamba-TasNet](https://github.com/xi-j/Mamba-TasNet)
- 适配器：`src/tibetan_ss/models/dual_path_mamba.py::DualPathMambaAdapter`
- 依赖：
  ```bash
  pip install speechbrain mamba-ssm causal-conv1d
  ```
  （`mamba-ssm` 需要 CUDA；CPU 目前不支持）
- 实现要点：复用 SpeechBrain 的 `Encoder/Decoder/Dual_Path_Model`，intra/inter 模块换成 Mamba-TasNet 仓库里的 `MambaBlocksSequential`。我们默认 8 kHz 配置（论文复现用）；切换到 16 kHz 需在 config 里改 `kernel_size=32, chunk_size=400`。

---

## MossFormer2

- 论文：Zhao et al., *MossFormer2: Combining Transformer and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation*, ICASSP 2024
- 上游：[speechbrain/speechbrain recipes/LibriMix](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriMix/separation)
- 适配器：`src/tibetan_ss/models/mossformer2.py::MossFormer2Adapter`
- 依赖：`pip install speechbrain`
- 实现要点：按 SpeechBrain 版本不同，MossFormer2 的 import path 会迁移，适配器会按序尝试已知路径。若都失败，请在 `_CANDIDATE_PATHS` 里补一个当前 SpeechBrain 版本对应的路径。变体通过 `variant: S|L` 选择。

---

## TIGER

- 论文：Xu et al., *TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation*, ICLR 2025
- 上游：[JusperLee/TIGER](https://github.com/JusperLee/TIGER)
- 适配器：`src/tibetan_ss/models/tiger.py::TIGERAdapter`
- 依赖：
  ```bash
  pip install rotary_embedding_torch asteroid-filterbanks
  ```
- 实现要点：直接 `from look2hear.models.tiger import TIGER`，默认使用论文中的 `TIGER-S` 超参（上游 `configs/speech-32k.yaml` 缩放到 16 kHz：`win=1024, stride=256, num_blocks=16`）。因为 TIGER 是 T-F 域模型，切换到 8 kHz 时推荐 `win=512, stride=128`。

---

## SepReformer

- 论文：Shin et al., *Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation*, NeurIPS 2024
- 上游：[dmlguq456/SepReformer](https://github.com/dmlguq456/SepReformer)
- 适配器：`src/tibetan_ss/models/sepreformer.py::SepReformerAdapter`
- 依赖：无额外 pip 依赖（上游代码纯 PyTorch）
- 实现要点：上游 `Model.forward` 返回 `(audio_list, audio_aux)`，我们把 `audio_list` stack 成 `(B, C, T)`；`audio_aux` 是各 stage 的中间输出，用来做深度监督。默认使用 `SepReformer_Base_WSJ0` 变体（8 kHz，最小，最快）；若要对齐 16 kHz + 混响场景，`variant: SepReformer_Large_DM_WHAMR`。

---

## DIP Frontend

- 论文：Wang et al., *Speech Separation With Pretrained Frontend to Minimize Domain Mismatch*, IEEE/ACM TASLP 2024
- 上游仓库只提供数据集代码；模型按论文描述复现
- 适配器 / 实现：`src/tibetan_ss/models/dip_frontend.py::DIPSeparator`
- 关键组件：
  - `SharedEncoder` — Conv + TCN × 3 作为 frontend
  - `_ProjectionHead` — 只在 SSL 预训练用的 BYOL projection
  - `target_frontend()` / `update_target()` — momentum-EMA 目标网络
  - `SharedMaskDecoder + WaveformDecoder + speaker_split(1×1 conv)` — 下游分离器
- 使用模式：
  1. （可选）先按 BYOL 对 frontend 做 SSL 预训练（见 SSL 训练 loop 的草稿 TODO，仓库预留了 `update_target` + `target_frontend` 的接口）
  2. 加载预训练 frontend 权重后，设 `freeze_frontend=true`，跑监督分离
  3. 或直接端到端训练（`freeze_frontend=false`）

---

## Proposed: Early-Separation

- 方案来源：`提出模型.docx`
- 实现：`src/tibetan_ss/models/proposed/` 子包
  - `encoder.py` — 共享主干编码器（Conv + TCN × 3）
  - `branch_head.py` — 独立双分支（TCN × 4 each）
  - `decoder.py` — 共享 mask 解码器（TCN × 3）+ Waveform 合成
  - `discriminator.py` — Multi-scale STFT PatchGAN（频谱域）
  - `losses.py` — 表示差异损失（Cosine + Orthogonal）+ hinge GAN
  - `model.py` — 组合
- 训练：`src/tibetan_ss/engine/gan_module.py::ProposedGANModule`，按 epoch 自动切换三阶段：
  - Stage 1：仅 `L_main = PIT + SI-SDR`
  - Stage 2：加 `L_rep` （`schedule.rep_from_epoch`）
  - Stage 3：加 `L_D` + hinge generator loss（`schedule.gan_from_epoch`）
- 关键设计决策：
  - **双分支对称破缺**：训练阶段在 branch 输入上各注入 `perturbation_std=1e-3` 的独立高斯噪声，防止两分支坍塌到同一解
  - **判别器频谱**：`n_ffts=[512, 1024, 2048]`，每尺度一个 `PatchGAN`，输入是 `log1p(|STFT|)`
  - **判别器更新顺序**：先 detach 的 fake 更新 D，再对 generator 反传 `-D(fake)` 的 hinge 项
  - **PIT 对齐**：判别器输入的 fake source 需按 PIT 最优排列后再送入 D（否则 D 学不到稳定的纯净度概念）
