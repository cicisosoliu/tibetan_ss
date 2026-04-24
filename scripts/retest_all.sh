#!/usr/bin/env bash
# ============================================================================
# Re-generate test mixtures (with fixed mixing.py) and re-evaluate all
# trained models WITHOUT re-training.
#
# Prerequisites:
#   - Models already trained (checkpoints exist under outputs/logs/<tag>/)
#   - pip install -e . done (so the fixed mixing.py is active)
#
# What this script does:
#   1. Re-run prepare_nict_tib1 (applies new minimum duration filter)
#   2. Regenerate ONLY the test split mixtures (train/val untouched)
#   3. For each model: clear old test_results/, find best checkpoint, run test
#   4. Aggregate results into outputs/summary_retest.md
#
# Usage:
#   scripts/retest_all.sh                        # all 6 models
#   scripts/retest_all.sh proposed_formal        # single model
#   scripts/retest_all.sh --skip-generate        # skip mixture regeneration
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# --- Parse args ---
SKIP_GENERATE=0
TARGET=""
for arg in "$@"; do
    case "$arg" in
        --skip-generate) SKIP_GENERATE=1 ;;
        *) TARGET="$arg" ;;
    esac
done

# --- Config paths ---
DATA_CONFIG="$REPO_ROOT/configs/data/nict_tib1_full.yaml"
LOG_ROOT="$REPO_ROOT/outputs/logs"

PASS=0
FAIL=0
ERRORS=""

# ============================================================================
# Step 1: Re-run data preparation (applies minimum duration filter)
# ============================================================================
if [ "$SKIP_GENERATE" = "0" ]; then
    echo ""
    echo "================================================================"
    echo " Step 1: Re-run prepare_nict_tib1 (apply audio file filter)"
    echo "================================================================"
    python -m tibetan_ss.data.scripts.prepare_nict_tib1 \
        --config "$DATA_CONFIG"

    echo ""
    echo "================================================================"
    echo " Step 2: Regenerate test mixtures (fixed mixing.py)"
    echo "================================================================"
    # Remove old test mixtures to avoid stale files remaining on disk
    TEST_MIX_DIR="$REPO_ROOT/data/mixtures/test"
    if [ -d "$TEST_MIX_DIR" ]; then
        echo "  Removing old test mixtures: $TEST_MIX_DIR"
        rm -rf "$TEST_MIX_DIR"
    fi

    python -m tibetan_ss.data.scripts.generate_mixtures \
        --config "$DATA_CONFIG" \
        --splits test --force

    echo ""
    echo "  ✓ Test mixtures regenerated with fixed full_length logic."
else
    echo ""
    echo "  ⊘ Skipping test mixture regeneration (--skip-generate)"
fi

# ============================================================================
# Step 3: Re-evaluate each model
# ============================================================================

find_best_ckpt() {
    local ckpt_dir="$1"
    if [ ! -d "$ckpt_dir" ]; then
        echo ""
        return
    fi
    # Prefer "best" checkpoint (highest si_sdri in filename)
    local best
    best=$(ls -1 "$ckpt_dir"/epoch*.ckpt 2>/dev/null | sort -t'i' -k3 -rn | head -1)
    if [ -n "$best" ]; then
        echo "$best"
        return
    fi
    # Fall back to last.ckpt
    if [ -f "$ckpt_dir/last.ckpt" ]; then
        echo "$ckpt_dir/last.ckpt"
        return
    fi
    echo ""
}

run_test() {
    local tag="$1"
    local config="$2"

    local save_dir="$LOG_ROOT/$tag"
    local ckpt_dir="$save_dir/checkpoints"
    local test_results_dir="$save_dir/test_results"

    echo ""
    echo "================================================================"
    echo " Re-test: $tag"
    echo "================================================================"

    # Find checkpoint
    local ckpt
    ckpt=$(find_best_ckpt "$ckpt_dir")
    if [ -z "$ckpt" ]; then
        echo "  ✗ No checkpoint found in $ckpt_dir — SKIP"
        FAIL=$((FAIL + 1))
        ERRORS="$ERRORS\n  - $tag: no checkpoint"
        return
    fi
    echo "  Checkpoint: $ckpt"

    # Clear old test results
    if [ -d "$test_results_dir" ]; then
        echo "  Clearing old test_results/ ..."
        rm -rf "$test_results_dir"
    fi

    # Run evaluation
    echo "  Running test ..."
    if python -m tibetan_ss.cli.evaluate \
        --config "$REPO_ROOT/configs/experiment/${config}.yaml" \
        --checkpoint "$ckpt" \
        --save-dir "$save_dir" \
        --max-audio 50 \
        --split test 2>&1; then
        echo "  ✓ $tag: test complete"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $tag: test FAILED"
        FAIL=$((FAIL + 1))
        ERRORS="$ERRORS\n  - $tag: test failed"
    fi
}

# --- Model definitions: tag → experiment config ---
if [ -n "$TARGET" ] && [ "$TARGET" != "--skip-generate" ]; then
    case "$TARGET" in
        proposed*)  run_test "proposed_formal"                  "proposed_formal" ;;
        tiger*)     run_test "baseline_tiger_nict"              "baseline_tiger_nict" ;;
        sep*)       run_test "baseline_sepreformer_nict"        "baseline_sepreformer_nict" ;;
        dp*|mamba*) run_test "baseline_dual_path_mamba_nict"    "baseline_dual_path_mamba_nict" ;;
        moss*)      run_test "baseline_mossformer2_nict"        "baseline_mossformer2_nict" ;;
        dip*)       run_test "ext_dip_nict"                     "ext_dip_nict" ;;
        *)          echo "Unknown model: $TARGET"; exit 1 ;;
    esac
else
    run_test "proposed_formal"                  "proposed_formal"
    run_test "baseline_tiger_nict"              "baseline_tiger_nict"
    run_test "baseline_sepreformer_nict"        "baseline_sepreformer_nict"
    run_test "baseline_dual_path_mamba_nict"    "baseline_dual_path_mamba_nict"
    run_test "baseline_mossformer2_nict"        "baseline_mossformer2_nict"
    run_test "ext_dip_nict"                     "ext_dip_nict"
fi

# ============================================================================
# Step 4: Aggregate results
# ============================================================================
echo ""
echo ">>> Aggregating results..."
python -m tibetan_ss.cli.aggregate_results \
    --root "$LOG_ROOT" \
    --output "$REPO_ROOT/outputs/summary_retest.md" || true

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================"
echo " Re-test results:  ✓ $PASS passed  |  ✗ $FAIL failed"
if [ $FAIL -gt 0 ]; then
    echo -e " Failures:$ERRORS"
fi
echo ""
echo " Summary: outputs/summary_retest.md"
echo " Per-model results: outputs/logs/<tag>/test_results/"
echo "================================================================"

exit $FAIL
