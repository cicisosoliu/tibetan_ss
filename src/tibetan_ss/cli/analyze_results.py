"""Post-hoc analysis of per-utterance test results.

Reads ``per_utterance.csv`` from each experiment and produces:

1. **Per-condition breakdown tables** (overlap bins, SNR bins, gender pair)
2. **Statistical significance tests** (paired Wilcoxon signed-rank between
   the proposed model and each baseline)
3. **LaTeX-ready tables** for direct copy into papers

Usage::

    PYTHONPATH=src python -m tibetan_ss.cli.analyze_results \\
        --root outputs/logs \\
        --proposed proposed_formal \\
        --output outputs/analysis/

Outputs:
    outputs/analysis/
    ├── breakdown_overlap.csv        # SI-SDRi by overlap bin per model
    ├── breakdown_snr.csv
    ├── breakdown_gender.csv
    ├── significance.csv             # p-values: proposed vs each baseline
    ├── main_table.tex               # LaTeX table for the paper
    └── analysis_report.md           # human-readable summary
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _load_per_utterance(run_dir: Path) -> list[dict] | None:
    csv_path = run_dir / "test_results" / "per_utterance.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            for k in ("si_sdr", "si_sdri", "pesq_wb", "stoi",
                       "overlap_ratio", "effective_overlap_ratio",
                       "snr_db", "level_diff_db", "length_samples"):
                if k in r:
                    try:
                        r[k] = float(r[k])
                    except (ValueError, TypeError):
                        r[k] = float("nan")
            rows.append(r)
    return rows


def _overlap_bin(ov: float) -> str:
    if np.isnan(ov):
        return "unknown"
    if ov < 0.3:
        return "low(0-0.3)"
    if ov < 0.7:
        return "mid(0.3-0.7)"
    return "high(0.7-1.0)"


def _snr_bin(snr: float) -> str:
    if np.isnan(snr):
        return "no_noise"
    if snr < 10:
        return "low(<10dB)"
    if snr < 20:
        return "mid(10-20dB)"
    return "high(>20dB)"


def _breakdown(all_data: dict[str, list[dict]], group_fn, group_name: str,
               metric: str = "si_sdri") -> list[dict]:
    """Build a table: rows=groups, columns=models."""
    groups: set[str] = set()
    model_groups: dict[str, dict[str, list[float]]] = {}
    for model, rows in all_data.items():
        model_groups[model] = {}
        for r in rows:
            g = group_fn(r)
            groups.add(g)
            model_groups[model].setdefault(g, []).append(r.get(metric, float("nan")))

    table = []
    for g in sorted(groups):
        row = {"condition": g}
        for model in all_data:
            vals = model_groups[model].get(g, [])
            row[model] = f"{np.nanmean(vals):.2f}" if vals else "—"
        table.append(row)
    return table


def _significance(proposed_rows: list[dict], baseline_rows: list[dict],
                  metric: str = "si_sdri") -> dict:
    """Paired Wilcoxon signed-rank test."""
    from scipy.stats import wilcoxon

    # Match by id
    base_map = {r["id"]: r[metric] for r in baseline_rows}
    paired_p, paired_b = [], []
    for r in proposed_rows:
        if r["id"] in base_map:
            pv = r[metric]
            bv = base_map[r["id"]]
            if np.isfinite(pv) and np.isfinite(bv):
                paired_p.append(pv)
                paired_b.append(bv)

    if len(paired_p) < 10:
        return {"n_paired": len(paired_p), "p_value": float("nan"),
                "mean_diff": float("nan"), "significant": False}

    diff = np.array(paired_p) - np.array(paired_b)
    stat, p = wilcoxon(diff)
    return {
        "n_paired": len(paired_p),
        "p_value": float(p),
        "mean_diff": float(diff.mean()),
        "significant": bool(p < 0.05),
    }


def _latex_table(all_data: dict[str, list[dict]], metrics=("si_sdri", "pesq_wb", "stoi")) -> str:
    """Generate a LaTeX tabular for the main results table."""
    cols = " ".join(f"c" for _ in metrics)
    lines = [
        r"\begin{tabular}{l" + cols + "}",
        r"\toprule",
        "Model & " + " & ".join(m.upper().replace("_", r"\_") for m in metrics) + r" \\",
        r"\midrule",
    ]
    for model, rows in all_data.items():
        vals = []
        for m in metrics:
            v = [r[m] for r in rows if np.isfinite(r.get(m, float("nan")))]
            vals.append(f"{np.mean(v):.2f}" if v else "—")
        lines.append(f"{model} & " + " & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/logs")
    p.add_argument("--proposed", default="proposed_formal", help="tag of proposed model")
    p.add_argument("--output", default="outputs/analysis")
    args = p.parse_args()

    root = Path(args.root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    all_data: dict[str, list[dict]] = {}
    for run_dir in sorted(root.glob("*")):
        if not run_dir.is_dir():
            continue
        rows = _load_per_utterance(run_dir)
        if rows:
            all_data[run_dir.name] = rows
            print(f"[analyze] loaded {run_dir.name}: {len(rows)} utterances")

    if not all_data:
        print("[analyze] no per_utterance.csv found under", root)
        return

    # Breakdown tables
    for name, group_fn in [
        ("overlap", lambda r: _overlap_bin(r.get("overlap_ratio", float("nan")))),
        ("snr", lambda r: _snr_bin(r.get("snr_db", float("nan")))),
        ("gender", lambda r: r.get("gender_pair", "unknown") or "unknown"),
    ]:
        table = _breakdown(all_data, group_fn, name)
        csv_path = out / f"breakdown_{name}.csv"
        if table:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=table[0].keys())
                writer.writeheader()
                writer.writerows(table)
            print(f"[analyze] wrote {csv_path}")

    # Significance tests
    proposed_tag = args.proposed
    if proposed_tag in all_data:
        sig_rows = []
        for model, rows in all_data.items():
            if model == proposed_tag:
                continue
            result = _significance(all_data[proposed_tag], rows)
            sig_rows.append({"baseline": model, **result})
        if sig_rows:
            csv_path = out / "significance.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=sig_rows[0].keys())
                writer.writeheader()
                writer.writerows(sig_rows)
            print(f"[analyze] wrote {csv_path}")
    else:
        print(f"[analyze] proposed tag '{proposed_tag}' not found — skipping significance tests")

    # LaTeX table
    tex = _latex_table(all_data)
    tex_path = out / "main_table.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"[analyze] wrote {tex_path}")

    # Markdown report
    report_lines = ["# Analysis Report\n"]
    report_lines.append(f"Models analyzed: {', '.join(all_data.keys())}\n")
    report_lines.append("## Overall Means\n")
    report_lines.append("| Model | SI-SDRi | PESQ-WB | STOI |")
    report_lines.append("| ----- | ------- | ------- | ---- |")
    for model, rows in all_data.items():
        sdri = np.nanmean([r["si_sdri"] for r in rows])
        pesq = np.nanmean([r["pesq_wb"] for r in rows])
        stoi = np.nanmean([r["stoi"] for r in rows])
        report_lines.append(f"| {model} | {sdri:.3f} | {pesq:.3f} | {stoi:.3f} |")
    report_lines.append("")

    if proposed_tag in all_data:
        report_lines.append("## Statistical Significance (Wilcoxon, proposed vs baseline)\n")
        report_lines.append("| Baseline | mean diff | p-value | sig? |")
        report_lines.append("| -------- | --------- | ------- | ---- |")
        for model in all_data:
            if model == proposed_tag:
                continue
            r = _significance(all_data[proposed_tag], all_data[model])
            sig = "yes" if r["significant"] else "no"
            report_lines.append(f"| {model} | {r['mean_diff']:.3f} | {r['p_value']:.4f} | {sig} |")

    report_path = out / "analysis_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[analyze] wrote {report_path}")


if __name__ == "__main__":
    main()
