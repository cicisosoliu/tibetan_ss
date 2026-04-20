#!/usr/bin/env bash
# ============================================================================
# 5× H100 80GB optimal launch: DDP across 5 GPUs, maximised batch + preload.
#
# Features:
#   - Per-model optimal batch/lr tuning for 5-GPU DDP
#   - Real-time ETA: after each model finishes, predicts remaining wall time
#   - Progress dashboard: elapsed / remaining / completed / total
#
# Usage:
#   scripts/run_all_5gpu.sh                   # all 6 models sequentially
#   scripts/run_all_5gpu.sh proposed_formal   # single model
#   scripts/run_all_5gpu.sh --dry             # dry-run: 1 epoch each
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

DRY=0
TARGET=""
for arg in "$@"; do
    if [ "$arg" = "--dry" ]; then DRY=1
    else TARGET="$arg"; fi
done

# --- Common 5-GPU overrides ---
GPU_COMMON=(
    training.trainer.devices=5
    training.trainer.strategy=ddp
    training.trainer.precision=bf16-mixed
    training.dataloader.num_workers=4
    training.dataloader.persistent_workers=true
    training.dataloader.pin_memory=true
    training.trainer.check_val_every_n_epoch=5
    training.scheduler.warmup_epochs=10
    data.preload_to_memory=true
    data.dynamic_mixing.cache_per_epoch=20000
    compile=false
)

if [ "$DRY" = "1" ]; then
    GPU_COMMON+=(
        training.trainer.max_epochs=1
        data.dynamic_mixing.cache_per_epoch=200
        training.eval_metrics="[si_sdr,si_sdri]"
        test_max_audio=3
        compile=false
    )
    echo "[5gpu] DRY RUN: 1 epoch per model (compile disabled)"
fi

# ---- Time tracking ----
TOTAL_START=$(date +%s)
MODEL_TIMES=()       # elapsed seconds per completed model
MODEL_NAMES=()       # names of completed models
MODELS_TOTAL=0       # will be set below
MODELS_DONE=0

fmt_duration() {
    local secs=$1
    local h=$((secs / 3600))
    local m=$(( (secs % 3600) / 60 ))
    local s=$((secs % 60))
    if [ $h -gt 0 ]; then
        printf "%dh%02dm%02ds" $h $m $s
    elif [ $m -gt 0 ]; then
        printf "%dm%02ds" $m $s
    else
        printf "%ds" $s
    fi
}

print_progress() {
    local now=$(date +%s)
    local elapsed=$((now - TOTAL_START))
    echo ""
    echo "┌─────────────────────────────────────────────────────────┐"
    printf "│  Progress: %d / %d models completed" "$MODELS_DONE" "$MODELS_TOTAL"
    printf "%*s│\n" $((27 - ${#MODELS_DONE} - ${#MODELS_TOTAL})) ""
    printf "│  Elapsed:  %-20s" "$(fmt_duration $elapsed)"
    printf "%*s│\n" 16 ""

    if [ $MODELS_DONE -gt 0 ] && [ $MODELS_DONE -lt $MODELS_TOTAL ]; then
        # Predict remaining: average time per model × remaining models
        local sum=0
        for t in "${MODEL_TIMES[@]}"; do sum=$((sum + t)); done
        local avg=$((sum / MODELS_DONE))
        local remaining=$(( avg * (MODELS_TOTAL - MODELS_DONE) ))
        local eta_epoch=$((now + remaining))
        local eta_str
        eta_str=$(date -d "@$eta_epoch" "+%H:%M:%S" 2>/dev/null || date -r "$eta_epoch" "+%H:%M:%S" 2>/dev/null || echo "?")

        printf "│  ETA:      %-20s" "$(fmt_duration $remaining)"
        printf "%*s│\n" 16 ""
        printf "│  Finish:   ~%-19s" "$eta_str"
        printf "%*s│\n" 16 ""
    fi

    if [ $MODELS_DONE -gt 0 ]; then
        echo "│                                                         │"
        echo "│  Per-model timings:                                     │"
        for i in $(seq 0 $((MODELS_DONE - 1))); do
            printf "│    %-30s %10s         │\n" "${MODEL_NAMES[$i]}" "$(fmt_duration ${MODEL_TIMES[$i]})"
        done
    fi
    echo "└─────────────────────────────────────────────────────────┘"
}

run_model() {
    local exp="$1" batch="$2" lr="$3"; shift 3
    local tag
    tag="$(grep '^tag:' "$REPO_ROOT/configs/experiment/${exp}.yaml" | awk '{print $2}')"
    echo ""
    echo "================================================================"
    echo " 🚀 $tag  (5×H100, batch=${batch}/gpu, eff_batch=$((batch * 5)), lr=${lr})"
    echo "    started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"

    local model_start=$(date +%s)

    "$REPO_ROOT/scripts/train.sh" "$REPO_ROOT/configs/experiment/${exp}.yaml" \
        training.dataloader.batch_size="$batch" \
        training.optimizer.lr="$lr" \
        _total_models="$MODELS_TOTAL" \
        _models_done="$MODELS_DONE" \
        "${GPU_COMMON[@]}" \
        "$@"

    local model_end=$(date +%s)
    local model_elapsed=$((model_end - model_start))
    MODEL_TIMES+=("$model_elapsed")
    MODEL_NAMES+=("$tag")
    MODELS_DONE=$((MODELS_DONE + 1))

    echo ""
    echo "  ✓ $tag finished in $(fmt_duration $model_elapsed)"
    print_progress
}

# --- Model definitions ---
run_proposed()      { run_model proposed_formal               16      3.0e-3  model.disc_lr=1.5e-3; }
run_tiger()         { run_model baseline_tiger_nict           12      3.0e-3  ; }
run_sepreformer()   { run_model baseline_sepreformer_nict     8       2.0e-3  ; }
run_dpmamba()       { run_model baseline_dual_path_mamba_nict 12      3.0e-3  ; }
run_mossformer2()   { run_model baseline_mossformer2_nict     8       2.0e-3  training.trainer.accumulate_grad_batches=2; }
run_dip()           { run_model ext_dip_nict                  16      3.0e-3  ; }

# --- Execute ---
if [ -n "$TARGET" ] && [ "$TARGET" != "--dry" ]; then
    MODELS_TOTAL=1
    case "$TARGET" in
        proposed*) run_proposed ;;
        tiger*)    run_tiger ;;
        sep*)      run_sepreformer ;;
        dp*|mamba*|dual*) run_dpmamba ;;
        moss*)     run_mossformer2 ;;
        dip*)      run_dip ;;
        *)         echo "Unknown model: $TARGET"; exit 1 ;;
    esac
else
    MODELS_TOTAL=6
    echo ""
    echo "================================================================"
    echo " 🏁 Starting 6-model training pipeline on 5× H100"
    echo "    $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    run_proposed
    run_tiger
    run_sepreformer
    run_dpmamba
    run_mossformer2
    run_dip
fi

# Aggregate
echo ""
echo ">>> Aggregating results..."
python -m tibetan_ss.cli.aggregate_results \
    --root "$REPO_ROOT/outputs/logs" \
    --output "$REPO_ROOT/outputs/summary.md"

# Final summary
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
echo ""
echo "================================================================"
echo " ✅ All done!"
echo "    Total wall time: $(fmt_duration $TOTAL_ELAPSED)"
echo "    Summary:         outputs/summary.md"
echo "    Started:         $(date -d "@$TOTAL_START" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r "$TOTAL_START" '+%Y-%m-%d %H:%M:%S')"
echo "    Finished:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
