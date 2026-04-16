#!/usr/bin/env bash
# Download and unpack the DEMAND noise corpus (16 kHz version) from Zenodo.
#
# https://zenodo.org/records/1227121
#
# After running this script you'll have:
#   $DEMAND_ROOT/
#   ├── DKITCHEN/{ch01.wav, ..., ch16.wav}
#   ├── DLIVING/...
#   ├── ...
#   └── TMETRO/...
#
# 17 categories are available at 16 kHz (SCAFE only ships at 48 kHz upstream –
# we skip it by default to avoid a sample-rate mismatch; pass
# --include-scafe to download the 48 k version and resample).
#
# Total on-disk footprint: ~1.5 GB zipped + ~2.5 GB extracted.
#
# Usage:
#   scripts/download_demand.sh [/optional/target/dir]
#
# If no target dir is given, uses $DEMAND_ROOT; if that's unset, defaults to
# $REPO_ROOT/DEMAND.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

TARGET="${1:-${DEMAND_ROOT:-$REPO_ROOT/DEMAND}}"
INCLUDE_SCAFE="${INCLUDE_SCAFE:-0}"          # set to 1 to also grab SCAFE (48k)

echo "==========================================================="
echo " DEMAND noise corpus downloader (16 kHz)"
echo " target:        $TARGET"
echo " include SCAFE: $INCLUDE_SCAFE"
echo "==========================================================="

mkdir -p "$TARGET"
cd "$TARGET"

CATEGORIES_16K=(
    DKITCHEN DLIVING DWASHING
    NFIELD NPARK NRIVER
    OHALLWAY OMEETING OOFFICE
    PCAFETER PRESTO PSTATION
    SPSQUARE STRAFFIC
    TBUS TCAR TMETRO
)

ZENODO_BASE="https://zenodo.org/records/1227121/files"

need_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[demand] required tool not found: $1" >&2
        exit 1
    fi
}
need_tool curl
need_tool unzip

download_one() {
    local name="$1"
    local zip_url="$2"
    local zip_path="${name}.zip"

    if [[ -d "$name" && -f "$name/ch01.wav" ]]; then
        echo "[demand] $name already extracted — skip"
        return 0
    fi

    if [[ ! -f "$zip_path" ]]; then
        echo "[demand] downloading $name …"
        # curl -C - resumes partial downloads; -L follows Zenodo's redirect.
        curl --fail --location --retry 3 --continue-at - \
             --output "$zip_path" "$zip_url"
    else
        echo "[demand] $zip_path already on disk — skipping download"
    fi

    echo "[demand] unpacking $name …"
    unzip -q -o "$zip_path"
    # Zenodo zips can extract into one of several layouts; normalise:
    #   DEMAND/<CATEGORY>/  ← desired
    # If the zip produced something else, move it into place.
    if [[ ! -d "$name" ]]; then
        # hunt for a dir that contains ch01.wav
        local found
        found="$(find . -maxdepth 3 -type f -name 'ch01.wav' -path "*/${name}/*" -print -quit || true)"
        if [[ -n "$found" ]]; then
            mv "$(dirname "$found")" "$name"
        else
            echo "[demand] WARN: could not locate ch01.wav for $name; inspect manually" >&2
        fi
    fi
}

for cat in "${CATEGORIES_16K[@]}"; do
    download_one "$cat" "${ZENODO_BASE}/${cat}_16k.zip?download=1"
done

if [[ "$INCLUDE_SCAFE" == "1" ]]; then
    echo "[demand] downloading SCAFE (only 48 kHz available; will need resampling)"
    download_one "SCAFE" "${ZENODO_BASE}/SCAFE_48k.zip?download=1"
    echo "[demand] NOTE: SCAFE is at 48 kHz. Resample to 16 kHz with:"
    echo "    for f in SCAFE/ch*.wav; do"
    echo "      sox \"\$f\" -r 16000 \"\${f%.wav}_16k.wav\" && mv \"\${f%.wav}_16k.wav\" \"\$f\""
    echo "    done"
fi

# cleanup zips (keep them? comment out if you want)
echo "[demand] removing zip files (extracted wavs preserved)"
rm -f ./*.zip

TOTAL_WAVS="$(find . -name '*.wav' | wc -l | tr -d ' ')"
echo "==========================================================="
echo " done — $TOTAL_WAVS wav files under $TARGET"
echo "==========================================================="
echo
echo " Next step: point the framework at this directory and regenerate."
echo "   export DEMAND_ROOT=\"$TARGET\""
echo "   # then re-run prepare + generate_mixtures (see docs/full_training.md)"
