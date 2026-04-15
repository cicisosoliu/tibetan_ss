"""Walk each experiment's output dir and build a Markdown comparison table.

Writes ``outputs/summary.md``:

| model | SI-SDR | SI-SDRi | PESQ-WB | STOI |
| ----- | ------ | ------- | ------- | ---- |
| ...   | ...    | ...     | ...     | ...  |
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("si_sdr", "si_sdri", "pesq_wb", "stoi")


def _load_latest_csv(run_dir: Path) -> dict[str, float] | None:
    csv_root = run_dir / "csv"
    if not csv_root.exists():
        # fall back to tensorboard dir – we don't parse that here
        return None
    versions = sorted(csv_root.glob("version_*"))
    if not versions:
        return None
    path = versions[-1] / "metrics.csv"
    if not path.exists():
        return None
    best = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for m in METRICS:
                key = f"test/{m}"
                if key in row and row[key]:
                    best[m] = float(row[key])
    return best or None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/logs", help="logger save root")
    p.add_argument("--output", default="outputs/summary.md")
    args = p.parse_args()

    root = Path(args.root)
    rows = []
    for run_dir in sorted(root.glob("*")):
        if not run_dir.is_dir():
            continue
        metrics = _load_latest_csv(run_dir) or {}
        rows.append((run_dir.name, metrics))

    lines = ["| model | SI-SDR | SI-SDRi | PESQ-WB | STOI |",
             "| ----- | ------ | ------- | ------- | ---- |"]
    for name, m in rows:
        lines.append(
            f"| {name} | {m.get('si_sdr', float('nan')):.3f} | "
            f"{m.get('si_sdri', float('nan')):.3f} | "
            f"{m.get('pesq_wb', float('nan')):.3f} | "
            f"{m.get('stoi', float('nan')):.3f} |"
        )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[aggregate] wrote {out}")


if __name__ == "__main__":
    main()
