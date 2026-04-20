# 论文数据收集指南

> 实验只跑一次成本很高。本文档说明框架在训练 + 测试过程中会自动收集哪些数据，
> 以及如何用分析脚本把这些数据转化为论文里的表格和图表。

## 1. 训练结束后的落盘数据

每个实验完成后，`outputs/logs/<tag>/` 下会有：

```
outputs/logs/<tag>/
├── resolved.yaml                              # 实验配置快照（完整可复现）
├── checkpoints/
│   ├── epoch040-sisdri13.420.ckpt             # top-3 best
│   └── last.ckpt
├── tb/version_0/events.out.tfevents.*         # TensorBoard 训练曲线
├── csv/version_0/metrics.csv                  # epoch 级均值（可选，取决于 logger）
├── complexity.json                            # 模型复杂度（需手动跑，见下）
└── test_results/                              # ← 新增！test 阶段自动生成
    ├── per_utterance.csv                      # 逐条测试结果
    ├── summary.json                           # 总体 + 按条件分桶的统计
    ├── audio/                                 # 前 50 条分离音频
    │   ├── <sample_id>/
    │   │   ├── mixture.wav
    │   │   ├── s1_est.wav
    │   │   ├── s2_est.wav
    │   │   ├── s1_ref.wav
    │   │   └── s2_ref.wav
    │   └── ...
    └── features/                              # 仅 proposed 模型
        ├── <sample_id>.pt                     # {z_a, z_b, features_a, features_b}
        └── ...
```

### 1.1 per_utterance.csv 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| id | str | 混合样本 ID（对应 manifest） |
| si_sdr | float | 该条的 SI-SDR (dB) |
| si_sdri | float | SI-SDR improvement (dB) |
| pesq_wb | float | PESQ-WB 分值 |
| stoi | float | STOI 分值 [0, 1] |
| overlap_ratio | float | 结构性重叠比（PPT 定义） |
| effective_overlap_ratio | float | 实际非静音重叠比 |
| snr_db | float | 混合时的 SNR (dB)；无噪声时为 NaN |
| gender_pair | str | MM / FF / FM |
| level_diff_db | float | 两路说话人的相对电平差 K (dB) |
| length_samples | int | 样本长度（采样点数） |

### 1.2 summary.json 内容

```json
{
  "n_samples": 3000,
  "si_sdri": {"mean": 13.42, "median": 13.85, "std": 4.21, "min": -2.1, "max": 25.3},
  "pesq_wb": {"mean": 3.02, ...},
  "stoi": {"mean": 0.89, ...},
  "si_sdri_by_overlap_bin": {
    "low(0-0.3)":  {"mean": 16.2, "n": 850},
    "mid(0.3-0.7)": {"mean": 12.8, "n": 1200},
    "high(0.7-1.0)": {"mean": 8.5, "n": 950}
  },
  "si_sdri_by_snr_bin": { ... },
  "si_sdri_by_gender_pair": { ... }
}
```

### 1.3 audio/ 目录

默认保存前 **50 条** test 样本的分离结果（可通过 `test_max_audio` 配置调整）。用于：
- 频谱图绘制
- 网页 demo
- MOS 主观评测

### 1.4 features/ 目录（仅 proposed）

每条 test 样本保存 `{z_a, z_b, features_a, features_b}` 的 tensor（`.pt` 文件），用于：
- t-SNE 可视化双分支表示分布
- 验证"早期分离"是否在编码层就产生了分化

## 2. 需要额外手动跑的脚本

### 2.1 模型复杂度（Params / MACs / RTF）

```bash
# 单个模型
PYTHONPATH=src python -m tibetan_ss.cli.model_complexity \
    --config configs/experiment/proposed_formal.yaml \
    --duration 3.0 --device cuda

# 全部 6 个模型
scripts/complexity_all.sh cuda
```

输出 `complexity.json`，包含参数量(M)、MACs(G/s)、推理时延(ms)、RTF。

### 2.2 分桶分析 + 显著性检验

```bash
PYTHONPATH=src python -m tibetan_ss.cli.analyze_results \
    --root outputs/logs \
    --proposed proposed_formal \
    --output outputs/analysis/
```

输出：
- `breakdown_overlap.csv` — 按重叠度分桶的 SI-SDRi
- `breakdown_snr.csv` — 按 SNR 分桶
- `breakdown_gender.csv` — 按性别组合分桶
- `significance.csv` — Wilcoxon 配对检验 p 值
- `main_table.tex` — 可直接嵌入 LaTeX 论文的结果表
- `analysis_report.md` — Markdown 汇总报告

### 2.3 可视化

```bash
# 频谱图对比
PYTHONPATH=src python -m tibetan_ss.cli.visualize spectrogram \
    --audio-dir outputs/logs/proposed_formal/test_results/audio/sample_0 \
    --output figures/spectrogram_sample0.pdf

# 训练曲线（所有模型在同一张图）
PYTHONPATH=src python -m tibetan_ss.cli.visualize curves \
    --root outputs/logs --metric val/si_sdri \
    --output figures/training_curves.pdf

# t-SNE 双分支表示
PYTHONPATH=src python -m tibetan_ss.cli.visualize tsne \
    --features-dir outputs/logs/proposed_formal/test_results/features \
    --output figures/tsne_branches.pdf
```

## 3. 论文各章节需要的数据对照

| 论文章节 | 需要的数据 | 来源 |
|---------|-----------|------|
| Table 1: 主对比表 | SI-SDRi / PESQ / STOI 均值 | `summary.json` 或 `analysis_report.md` |
| Table 2: 复杂度对比 | Params / MACs / RTF | `complexity.json` |
| Table 3: 消融实验 | 各消融变体的指标 | 多次跑 proposed + 不同 config |
| Table 4: 按条件分桶 | overlap/SNR/gender 分组指标 | `breakdown_*.csv` |
| Figure: 训练曲线 | val/si_sdri vs epoch | `visualize curves` |
| Figure: 频谱图 | mixture/est/ref 的 STFT | `visualize spectrogram` |
| Figure: t-SNE | 双分支表示分布 | `visualize tsne` |
| 正文: 统计显著性 | p-value < 0.05 | `significance.csv` |
| 补充材料: 音频 demo | 分离前后 wav | `test_results/audio/` |
| LaTeX 主表 | 可直接 \input 的表格 | `main_table.tex` |

## 4. 完整的"跑一次实验 → 出论文数据"流程

```bash
# 1. 训练（自动落盘 per_utterance.csv + audio + features）
scripts/run_all.sh nict

# 2. 复杂度统计
scripts/complexity_all.sh cuda

# 3. 分析（分桶 + 显著性 + LaTeX 表）
PYTHONPATH=src python -m tibetan_ss.cli.analyze_results \
    --root outputs/logs --proposed proposed_formal --output outputs/analysis/

# 4. 可视化
PYTHONPATH=src python -m tibetan_ss.cli.visualize curves \
    --root outputs/logs --output figures/training_curves.pdf
PYTHONPATH=src python -m tibetan_ss.cli.visualize tsne \
    --features-dir outputs/logs/proposed_formal/test_results/features \
    --output figures/tsne_branches.pdf
# 挑几个有代表性的样本画频谱图
for sid in sample_0 sample_5 sample_10; do
    PYTHONPATH=src python -m tibetan_ss.cli.visualize spectrogram \
        --audio-dir outputs/logs/proposed_formal/test_results/audio/$sid \
        --output figures/spectrogram_${sid}.pdf
done

# 5. 汇总表
PYTHONPATH=src python -m tibetan_ss.cli.aggregate_results \
    --root outputs/logs --output outputs/summary.md
```

## 5. 配置说明

| 参数 | 位置 | 默认 | 说明 |
|------|------|------|------|
| `test_max_audio` | experiment YAML 顶层 | 50 | test 时保存多少条分离音频 |
| `save_features` | GAN module 自动开启 | proposed=true, 其它=false | 是否保存 z_a/z_b |

增加保存音频数量：

```bash
scripts/train.sh configs/experiment/proposed_formal.yaml test_max_audio=200
```
