# Data Pipeline

> 对齐 PPT『藏语多人语音分离20260414.pptx』第 9-12 页的数据规划，以及
> `training_data_generation.md` 里综述的 LibriMix 合成范式。

## 1. 数据源

| 用途 | 语料 | 内容 |
|------|------|------|
| 藏语（干净单人） | **NICT-Tib1**，拉萨方言，8 男 12 女，共 33.5 h | 一人一目录，子目录含若干 .wav |
| 背景噪声 | **DEMAND** | 6 类环境 × 若干文件（家庭/办公室/公共/交通/自然/街道） |
| （可选）混响 | 任意 RIR 语料（WHAMR! / FUSS / Matterport3D），不默认启用 |

## 2. 说话人划分（reproducible）

按 PPT 的比例 **train 6♂+8♀ / val 1♂+2♀ / test 1♂+2♀**，每个划分内部按声纹相似度做过聚类（这一聚类步骤由上游准备脚本的 Speaker ID 正则 `--speaker-regex "^(?P<gender>[MF])(?P<sid>\d+)"` 解析，默认基于文件名；若文件名不遵循该规范，可以覆盖）。

`prepare_nict_tib1.py` 会产出：

```
<output_root>/manifests/
├── speakers_train.json        # [{id, gender, files: [...]}, ...]
├── speakers_val.json
├── speakers_test.json
├── all_speakers.json          # 并集（含 split 字段）
├── noise_train.json           # DEMAND 文件按 80/10/10 随机划分
├── noise_val.json
└── noise_test.json
```

**种子固定**：`configs/data/default.yaml::speaker_split.seed = 20260415`，保证任何运行产出的划分完全一致。

## 3. 混合生成

每条混合样本的生成流程（严格对齐 PPT 第 12 页）：

1. **挑选两位不同的说话人**（按 `gender_pairing` 规则，默认 `random`）
2. **裁剪**：train/val 从 `U(2,4)s` 里采样一个长度；test 不裁剪
3. **RMS 归一化**：两路语音归到共同的 -25 dBFS 参考
4. **相对电平 K**：从 `U(-5, +5) dB` 采样，一正一负作用到两路
5. **Overlap 放置**：按 `U(0,1)` / 混合分布（val 80%+20%, test 70%+30%）随机采 overlap 比，布局 `[A-only | overlap | B-only]`，左右顺序随机
6. **混音**：`mix = s1 + s2`
7. **（可选）噪声**：从 DEMAND 挑一条噪声段，按 `U(2.5, 30) dB` (train/val) / `U(0, 30) dB` (test) 的整体 SNR 缩放后相加
8. **防裁剪**：若峰值 > 0.99，按比例整体衰减

每条样本的 metadata 会被写入 manifest，字段包括：
`overlap_ratio, level_diff_db, level_db_a, level_db_b, snr_db, gender_pair,
segment_samples, segment_seconds`。

### 3.1 离线预生成模式（默认）

```bash
scripts/prepare_data.sh configs/data/sr16k.yaml
```

生成后目录结构：

```
<output_root>/mixtures/
├── train/<mix_id>/{mixture,s1,s2[,noise]}.wav
├── val/...
└── test/...
<output_root>/manifests/
├── train.json        # {split, items[{id, mixture_path, source_paths, meta, ...}], sample_rate}
├── val.json
└── test.json
```

Mix ID 遵循 PPT 第 11 页示例：`mix_train_MM_ov084_n1_ns214_rlp12_011036`（gender + overlap*100 + noise标记 + SNR×10 + K×10 + idx）。

### 3.2 动态混合（可选）

打开 `configs/data/default.yaml::dynamic_mixing.enabled=true`，train split 会切到 `TibetanMixDataset(dynamic=True)`，即每次 `__getitem__` 现场采样+合成，不读预生成的 manifest。val/test 仍走离线 manifest，保证公平评估。

动态混合在 MossFormer2、TF-Locoformer、TIGER 中都是 SOTA 配置；若显存允许建议开启以获得额外 ~1 dB。

### 3.3 混响 Hook（预留）

```yaml
reverb:
  enabled: false           # 默认关闭
  rir_root: /path/to/rirs  # 每个 RIR 一个 .wav 文件
  t60_range: [0.2, 0.8]
```

开启后会在 mixing 的第 3 步前对每路源卷积一个采样的 RIR。当前项目默认不启用，以对齐 PPT 规划中不含混响的配置。

## 4. Lengths 与 collate

- train/val：每条都是 `L = U(2,4)s` 的固定长，直接堆 batch
- test：片段长度可变，DataLoader 的 `collate_fn=collate_variable_length` 做右侧 zero-pad 并返回 `length` 张量；`test_step` 中按 `length` 对齐评估

## 5. 重采样与通道

- `sample_rate` 在 config 中统一控制（16 kHz 默认 / 8 kHz 可选），`read_audio` 用 `librosa.resample` 做到目标 SR
- 所有源音频、噪声统一降为单通道（均值通道）

## 6. 质量检查

每次 `scripts/prepare_data.sh` 运行后，建议：

```bash
python -c "
import json, glob, numpy as np
for split in ['train','val','test']:
    with open(f'$TIBETAN_SS_OUTPUT/manifests/{split}.json') as f: d = json.load(f)
    ov = [it['meta']['overlap_ratio'] for it in d['items']]
    K  = [it['meta']['level_diff_db'] for it in d['items']]
    snr = [it['meta']['snr_db'] for it in d['items'] if it['meta']['snr_db'] is not None]
    print(split, 'N=', len(d['items']),
          'ov mean/std=', np.mean(ov), np.std(ov),
          'K std=', np.std(K),
          'snr mean=', np.mean(snr))
"
```

典型输出（供对照）：

```
train N= 20000 ov mean/std= 0.50 0.29 K std= 2.9 snr mean= 16.3
val   N=  3000 ov mean/std= 0.54 0.31 K std= 2.9 snr mean= 16.3
test  N=  3000 ov mean/std= 0.56 0.31 K std= 2.9 snr mean= 15.0
```
