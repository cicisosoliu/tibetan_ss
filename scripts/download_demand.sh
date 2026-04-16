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
# we skip it by default; set INCLUDE_SCAFE=1 to also grab the 48k version).
#
# Total on-disk footprint: ~1.5 GB zipped + ~2.5 GB extracted.
#
# Usage:
#   scripts/download_demand.sh [/optional/target/dir]
#
# Network-aware behaviour (safe against flaky connections):
#   * curl --continue-at - resumes any partial .zip on disk
#   * --speed-time / --speed-limit aborts stalled connections
#   * each category gets MAX_ATTEMPTS outer retries
#   * every .zip is integrity-checked with `unzip -t` before we trust it;
#     truncated/garbled files are re-fetched automatically
#
# Tunables (export as env vars to override):
#   MAX_ATTEMPTS=5      outer retries per file
#   RETRY_DELAY=10      seconds between outer retries
#   SPEED_TIME=30       seconds of low throughput before curl gives up
#   SPEED_LIMIT=1024    bytes/sec – below this the connection is "stalled"
#   KEEP_ZIPS=1         keep .zip files after extraction (default 1)
#   INCLUDE_SCAFE=0     also fetch SCAFE_48k.zip (default 0)
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

MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"           # outer retries per category
RETRY_DELAY="${RETRY_DELAY:-10}"            # seconds between outer retries
SPEED_TIME="${SPEED_TIME:-30}"              # abort if <SPEED_LIMIT B/s for this many seconds
SPEED_LIMIT="${SPEED_LIMIT:-1024}"          # minimum acceptable bytes/sec

# ---------------------------------------------------------------------------
# Return 0 if the local zip passes `unzip -t`, non-zero otherwise.
# We also treat an empty / tiny file as corrupt so we don't waste curl calls
# trying to "resume" a 0-byte file with a server that mis-handles range.
# ---------------------------------------------------------------------------
zip_is_valid() {
    local f="$1"
    [[ -s "$f" ]] || return 1
    # --test is quiet and exits non-zero on any structural problem.
    unzip -tq "$f" >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Fetch a Content-Length for the remote URL (follows redirects). Prints the
# value on stdout; empty string if the server doesn't report it or on error.
# ---------------------------------------------------------------------------
remote_size() {
    local url="$1"
    local size
    size="$(curl --silent --location --head \
                 --write-out '%{size_download}\n%{header_json}' \
                 --output /dev/null "$url" 2>/dev/null \
            | sed -n 's/.*[Cc]ontent-[Ll]ength": *"\?\([0-9][0-9]*\)"\?.*/\1/p' \
            | tail -n1)"
    echo "${size:-}"
}

download_one() {
    local name="$1"
    local zip_url="$2"
    local zip_path="${name}.zip"

    if [[ -d "$name" && -f "$name/ch01.wav" ]]; then
        echo "[demand] $name already extracted — skip"
        return 0
    fi

    # Fast-path: if we've already got a valid zip on disk, just go to unpack.
    if [[ -f "$zip_path" ]] && zip_is_valid "$zip_path"; then
        echo "[demand] $zip_path exists and passes integrity check — skip download"
    else
        # Retry loop: each attempt uses `curl -C -` to resume partial content.
        # HTTP 416 (range not satisfiable — file already full) is treated as a
        # success provided `unzip -t` agrees.
        local attempt rc
        for (( attempt = 1; attempt <= MAX_ATTEMPTS; attempt++ )); do
            if [[ -f "$zip_path" ]]; then
                echo "[demand] resuming $name (attempt $attempt/$MAX_ATTEMPTS, $(du -h "$zip_path" | cut -f1) already on disk) …"
            else
                echo "[demand] downloading $name (attempt $attempt/$MAX_ATTEMPTS) …"
            fi

            rc=0
            curl --fail --location \
                 --retry 3 --retry-delay 5 \
                 --speed-time "$SPEED_TIME" --speed-limit "$SPEED_LIMIT" \
                 --continue-at - \
                 --output "$zip_path" "$zip_url" \
                 || rc=$?

            # rc=22 == HTTP error from server. Most common cause during
            # resume: 416 "Range Not Satisfiable" when local file is already
            # complete. If the zip passes integrity check, we're done.
            if [[ $rc -eq 0 ]] && zip_is_valid "$zip_path"; then
                break
            fi
            if [[ $rc -eq 22 ]] && zip_is_valid "$zip_path"; then
                echo "[demand] server returned 416 but local file is complete — OK"
                break
            fi

            # The download ended but the file is corrupt/truncated. Decide
            # whether to keep attempting resume or start over.
            if [[ -f "$zip_path" ]] && ! zip_is_valid "$zip_path"; then
                local local_size remote_len
                local_size="$(stat -c %s "$zip_path" 2>/dev/null || stat -f %z "$zip_path" 2>/dev/null || echo 0)"
                remote_len="$(remote_size "$zip_url")"
                if [[ -n "$remote_len" ]] && [[ "$local_size" -gt "$remote_len" ]]; then
                    echo "[demand] local file ($local_size B) is larger than remote ($remote_len B) — starting over"
                    rm -f "$zip_path"
                fi
            fi

            echo "[demand] attempt $attempt failed (curl rc=$rc); waiting ${RETRY_DELAY}s …" >&2
            sleep "$RETRY_DELAY"
        done

        if ! zip_is_valid "$zip_path"; then
            echo "[demand] ERROR: $zip_path still corrupt after $MAX_ATTEMPTS attempts." >&2
            echo "         Delete it manually and re-run:  rm '$(pwd)/$zip_path'" >&2
            return 1
        fi
    fi

    echo "[demand] unpacking $name …"
    unzip -q -o "$zip_path"
    # Zenodo zips can extract into one of several layouts; normalise:
    #   DEMAND/<CATEGORY>/  ← desired
    # If the zip produced something else, move it into place.
    if [[ ! -d "$name" ]]; then
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

KEEP_ZIPS="${KEEP_ZIPS:-1}"
if [[ "$KEEP_ZIPS" != "1" ]]; then
    echo "[demand] removing zip files (extracted wavs preserved)"
    rm -f ./*.zip
else
    echo "[demand] keeping .zip files on disk — set KEEP_ZIPS=0 to delete"
fi

TOTAL_WAVS="$(find . -name '*.wav' | wc -l | tr -d ' ')"
echo "==========================================================="
echo " done — $TOTAL_WAVS wav files under $TARGET"
echo "==========================================================="
echo
echo " Next step: point the framework at this directory and regenerate."
echo "   export DEMAND_ROOT=\"$TARGET\""
echo "   # then re-run prepare + generate_mixtures (see docs/full_training.md)"
