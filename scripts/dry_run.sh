#!/usr/bin/env bash
# Dry-run: 每个模型跑 1 epoch（小数据）→ test → 分析 → 可视化 → 验证全流程
# 通过后自动清理所有落盘内容。
#
# Usage:
#   scripts/dry_run.sh              # 默认用 GPU
#   scripts/dry_run.sh cpu          # CPU 模式（无 GPU 时）
#   KEEP=1 scripts/dry_run.sh      # 跑完不删除，保留检查
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

DEVICE="${1:-auto}"
KEEP="${KEEP:-0}"
DRY_TAG="_dryrun"
DRY_OUTPUT="$REPO_ROOT/outputs/logs_dryrun"
DRY_DATA="$REPO_ROOT/data_dryrun"
DRY_FIGURES="$REPO_ROOT/figures_dryrun"
DRY_ANALYSIS="$REPO_ROOT/outputs/analysis_dryrun"

: "${TIBETAN_ROOT:=$REPO_ROOT/Tibetan}"
: "${DEMAND_ROOT:=}"
export TIBETAN_ROOT DEMAND_ROOT

PASS=0
FAIL=0
SKIP=0
ERRORS=""

run_step() {
    local desc="$1"; shift
    echo ""
    echo "━━━ $desc ━━━"
    if "$@" 2>&1; then
        echo "  ✓ $desc"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $desc FAILED (rc=$?)"
        FAIL=$((FAIL + 1))
        ERRORS="$ERRORS\n  - $desc"
    fi
}

skip_step() {
    echo "  ⊘ $1 (skipped: $2)"
    SKIP=$((SKIP + 1))
}

echo "==========================================================="
echo " Dry-run: full pipeline validation (1 epoch per model)"
echo " TIBETAN_ROOT = $TIBETAN_ROOT"
echo " DRY_OUTPUT   = $DRY_OUTPUT"
echo " DEVICE       = $DEVICE"
echo "==========================================================="

# ------------------------------------------------------------------
# Step 1: Prepare tiny data (reuse existing manifests if possible)
# ------------------------------------------------------------------
run_step "prepare data (nict_tib1, 50 val + 50 test)" \
    python -m tibetan_ss.data.scripts.prepare_nict_tib1 \
        --config "$REPO_ROOT/configs/data/nict_tib1.yaml" \
        --override-output-root "$DRY_DATA"

run_step "generate tiny mixtures" \
    python -m tibetan_ss.data.scripts.generate_mixtures \
        --config "$REPO_ROOT/configs/data/nict_tib1.yaml" \
        --override-output-root "$DRY_DATA" \
        --splits train val test --force

# ------------------------------------------------------------------
# Step 2: Train each model for 1 epoch
# ------------------------------------------------------------------
# Common overrides: 1 epoch, tiny DM cache, small batch, csv logger, dry output
COMMON_OVERRIDES=(
    training.trainer.max_epochs=1
    training.trainer.precision=32-true
    training.trainer.accelerator="$DEVICE"
    training.dataloader.batch_size=2
    training.dataloader.num_workers=0
    training.dataloader.persistent_workers=false
    training.early_stop.enabled=false
    training.logger.name=csv
    training.logger.save_dir="$DRY_OUTPUT"
    training.trainer.check_val_every_n_epoch=1
    training.eval_metrics="[si_sdr,si_sdri]"
    data.paths.output_root="$DRY_DATA"
    data.dynamic_mixing.cache_per_epoch=100
    data.dynamic_mixing.enabled=true
    data.preload_to_memory=true
    data.offline.num_mixtures.val=50
    data.offline.num_mixtures.test=50
    compile=false
    test_max_audio=5
)

# --- Proposed (GAN engine) ---
run_step "train proposed (1 epoch)" \
    python -m tibetan_ss.cli.train \
        --config "$REPO_ROOT/configs/experiment/smoke_proposed.yaml" \
        tag="proposed${DRY_TAG}" \
        model.schedule.rep_from_epoch=0 \
        model.schedule.gan_from_epoch=0 \
        "${COMMON_OVERRIDES[@]}"

# --- TIGER ---
run_step "train TIGER (1 epoch)" \
    python -m tibetan_ss.cli.train \
        --config "$REPO_ROOT/configs/experiment/baseline_tiger_nict.yaml" \
        tag="tiger${DRY_TAG}" \
        "${COMMON_OVERRIDES[@]}"

# --- SepReformer ---
run_step "train SepReformer (1 epoch)" \
    python -m tibetan_ss.cli.train \
        --config "$REPO_ROOT/configs/experiment/baseline_sepreformer_nict.yaml" \
        tag="sepreformer${DRY_TAG}" \
        "${COMMON_OVERRIDES[@]}"

# --- DIP Frontend ---
run_step "train DIP Frontend (1 epoch)" \
    python -m tibetan_ss.cli.train \
        --config "$REPO_ROOT/configs/experiment/ext_dip_nict.yaml" \
        tag="dip${DRY_TAG}" \
        "${COMMON_OVERRIDES[@]}"

# --- Dual-path Mamba (needs CUDA + mamba-ssm) ---
if python -c "import mamba_ssm" 2>/dev/null && [ "$DEVICE" != "cpu" ]; then
    run_step "train Dual-path Mamba (1 epoch)" \
        python -m tibetan_ss.cli.train \
            --config "$REPO_ROOT/configs/experiment/baseline_dual_path_mamba_nict.yaml" \
            tag="dpmamba${DRY_TAG}" \
            "${COMMON_OVERRIDES[@]}"
else
    skip_step "Dual-path Mamba" "mamba-ssm not installed or CPU mode"
fi

# --- MossFormer2 (needs MossFormer2 clone) ---
if [ -d "$REPO_ROOT/third_party/MossFormer2" ]; then
    run_step "train MossFormer2 (1 epoch)" \
        python -m tibetan_ss.cli.train \
            --config "$REPO_ROOT/configs/experiment/baseline_mossformer2_nict.yaml" \
            tag="mossformer2${DRY_TAG}" \
            "${COMMON_OVERRIDES[@]}"
else
    skip_step "MossFormer2" "third_party/MossFormer2 not cloned"
fi

# ------------------------------------------------------------------
# Step 3: Verify test_results were produced
# ------------------------------------------------------------------
echo ""
echo "━━━ Checking test_results ━━━"
ALL_OK=true
for tag in proposed tiger sepreformer dip; do
    tag="${tag}${DRY_TAG}"
    csv_file="$DRY_OUTPUT/${tag}/test_results/per_utterance.csv"
    audio_dir="$DRY_OUTPUT/${tag}/test_results/audio"
    if [ -f "$csv_file" ]; then
        rows=$(wc -l < "$csv_file")
        echo "  ✓ $tag: per_utterance.csv ($rows rows)"
    else
        echo "  ✗ $tag: per_utterance.csv MISSING"
        ALL_OK=false
    fi
    if [ -d "$audio_dir" ]; then
        n_audio=$(find "$audio_dir" -name "mixture.wav" | wc -l)
        echo "    audio: $n_audio samples saved"
    fi
done

# Proposed features
feat_dir="$DRY_OUTPUT/proposed${DRY_TAG}/test_results/features"
if [ -d "$feat_dir" ]; then
    n_feat=$(find "$feat_dir" -name "*.pt" | wc -l)
    echo "  ✓ proposed features: $n_feat .pt files (for t-SNE)"
fi

# ------------------------------------------------------------------
# Step 4: Run analysis scripts
# ------------------------------------------------------------------
run_step "aggregate results" \
    python -m tibetan_ss.cli.aggregate_results \
        --root "$DRY_OUTPUT" --output "$DRY_OUTPUT/summary.md"

run_step "analyze results (breakdown + significance)" \
    python -m tibetan_ss.cli.analyze_results \
        --root "$DRY_OUTPUT" \
        --proposed "proposed${DRY_TAG}" \
        --output "$DRY_ANALYSIS"

# ------------------------------------------------------------------
# Step 5: Run visualization (if audio exists)
# ------------------------------------------------------------------
sample_dir="$DRY_OUTPUT/proposed${DRY_TAG}/test_results/audio"
first_sample=$(find "$sample_dir" -maxdepth 1 -type d | head -2 | tail -1)
if [ -n "$first_sample" ] && [ "$first_sample" != "$sample_dir" ]; then
    run_step "visualize spectrogram" \
        python -m tibetan_ss.cli.visualize spectrogram \
            --audio-dir "$first_sample" \
            --output "$DRY_FIGURES/spectrogram.png"
fi

if [ -d "$feat_dir" ]; then
    run_step "visualize t-SNE" \
        python -m tibetan_ss.cli.visualize tsne \
            --features-dir "$feat_dir" \
            --max-samples 20 \
            --output "$DRY_FIGURES/tsne.png"
fi

run_step "visualize training curves" \
    python -m tibetan_ss.cli.visualize curves \
        --root "$DRY_OUTPUT" --metric val/si_sdri \
        --output "$DRY_FIGURES/curves.png"

# ------------------------------------------------------------------
# Step 6: Model complexity (quick, cpu-safe)
# ------------------------------------------------------------------
run_step "model complexity (proposed)" \
    python -m tibetan_ss.cli.model_complexity \
        --config "$REPO_ROOT/configs/experiment/smoke_proposed.yaml" \
        --duration 1.0 --device cpu \
        --output "$DRY_OUTPUT/proposed${DRY_TAG}/complexity.json"

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "==========================================================="
echo " Dry-run results:  ✓ $PASS passed  |  ✗ $FAIL failed  |  ⊘ $SKIP skipped"
if [ $FAIL -gt 0 ]; then
    echo -e " Failures:$ERRORS"
fi
echo "==========================================================="

# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------
if [ "$KEEP" = "0" ] && [ $FAIL -eq 0 ]; then
    echo ""
    echo ">>> Cleaning up dry-run artifacts..."
    rm -rf "$DRY_OUTPUT" "$DRY_DATA" "$DRY_FIGURES" "$DRY_ANALYSIS"
    echo ">>> Cleaned. All dry-run data removed."
elif [ "$KEEP" = "0" ] && [ $FAIL -gt 0 ]; then
    echo ""
    echo ">>> NOT cleaning up (there were failures). Inspect:"
    echo "    $DRY_OUTPUT"
    echo "    $DRY_DATA"
    echo "    To manually clean: rm -rf $DRY_OUTPUT $DRY_DATA $DRY_FIGURES $DRY_ANALYSIS"
else
    echo ""
    echo ">>> KEEP=1: dry-run data preserved at:"
    echo "    $DRY_OUTPUT"
fi

exit $FAIL
