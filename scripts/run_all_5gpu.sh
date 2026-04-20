#!/usr/bin/env bash
# ============================================================================
# 5× H100 80GB optimal launch: DDP across 5 GPUs, maximised batch + preload.
#
# Per-model tuning rationale:
#   effective_batch = batch_per_gpu × 5
#   lr = base_lr × sqrt(effective_batch / base_batch)  (sqrt scaling rule)
#   warmup = 10 epochs (to absorb large-batch instability)
#   num_workers = 4 per process (preload makes I/O free; workers only mix)
#   preload_to_memory = true (each DDP process loads ~13 GB; 5×13=65 GB RAM)
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
)

if [ "$DRY" = "1" ]; then
    GPU_COMMON+=(
        training.trainer.max_epochs=1
        data.dynamic_mixing.cache_per_epoch=200
        training.eval_metrics="[si_sdr,si_sdri]"
        test_max_audio=3
    )
    echo "[5gpu] DRY RUN: 1 epoch per model"
fi

# --- Per-model settings ---
# Format: experiment_yaml  batch_per_gpu  lr  extra_overrides...

run_model() {
    local exp="$1" batch="$2" lr="$3"; shift 3
    local tag
    tag="$(grep '^tag:' "$REPO_ROOT/configs/experiment/${exp}.yaml" | awk '{print $2}')"
    echo ""
    echo "================================================================"
    echo " 🚀 $tag  (5×H100, batch=${batch}/gpu, lr=${lr})"
    echo "================================================================"
    "$REPO_ROOT/scripts/train.sh" "$REPO_ROOT/configs/experiment/${exp}.yaml" \
        training.dataloader.batch_size="$batch" \
        training.optimizer.lr="$lr" \
        "${GPU_COMMON[@]}" \
        "$@"
}

# --- Model definitions ---
#                     experiment config                     batch/gpu  lr      extra
run_proposed()      { run_model proposed_formal               16      3.0e-3  model.disc_lr=1.5e-3; }
run_tiger()         { run_model baseline_tiger_nict           12      3.0e-3  ; }
run_sepreformer()   { run_model baseline_sepreformer_nict     8       2.0e-3  ; }
run_dpmamba()       { run_model baseline_dual_path_mamba_nict 12      3.0e-3  ; }
run_mossformer2()   { run_model baseline_mossformer2_nict     8       2.0e-3  training.trainer.accumulate_grad_batches=2; }
run_dip()           { run_model ext_dip_nict                  16      3.0e-3  ; }

# --- Execute ---
if [ -n "$TARGET" ] && [ "$TARGET" != "--dry" ]; then
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

echo ""
echo "================================================================"
echo " All done. Summary: outputs/summary.md"
echo "================================================================"
