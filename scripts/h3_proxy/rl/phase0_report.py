"""Phase-0 report: is there headroom across seeds, and does picking by a reward find it?

Reads ``phase0_score.py`` rows. For every reward it reports how much of its variance lies
between seeds of one sample (the headroom a fine-tune could exploit), its within-sample rank
correlation with each independent metric, and the gain in those metrics when the best seed by
the reward is kept instead of a random one (bootstrap 95% interval over samples).

    python phase0_report.py scores.jsonl --out report.md --json report.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REWARDS = ("depth", "motion", "vlm_event", "vlm_identity", "vlm_quality")
METRICS = ("psnr_late", "ssim_late", "dino")
COMPOSITE = "composite"


def ranks(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="stable")
    out = np.empty(len(values))
    out[order] = np.arange(len(values))
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def bootstrap(values: np.ndarray, rounds: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(rounds, len(values)), replace=True).mean(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("source", "serve") == "serve":
            by_sample[row["sample"]].append(row)
    return {s: sorted(r, key=lambda x: x["seed"]) for s, r in by_sample.items() if len(r) >= 2}


def with_composite(groups: dict[str, list[dict]]) -> list[str]:
    """Adds the within-sample z-score sum of the available rewards; returns the rewards present."""
    present = [r for r in REWARDS if any(row.get(r) is not None for rows in groups.values() for row in rows)]
    for rows in groups.values():
        total = np.zeros(len(rows))
        for reward in present:
            values = np.array([row.get(reward) for row in rows], dtype=float)
            if np.isnan(values).any():
                continue
            spread = values.std()
            if spread > 0:
                total += (values - values.mean()) / spread
        for row, value in zip(rows, total):
            row[COMPOSITE] = float(value)
    return present + [COMPOSITE]


def analyse(rows: list[dict]) -> dict:
    groups = grouped(rows)
    selectors = with_composite(groups)
    report: dict = {"samples": len(groups), "seeds": sorted({r["seed"] for rs in groups.values() for r in rs})}

    evals = {r["sample"]: r for r in rows if r.get("source") == "eval"}
    pairs = [(evals[s]["psnr_late"], next(r for r in rs if r["seed"] == 42)["psnr_late"])
             for s, rs in groups.items() if s in evals and any(r["seed"] == 42 for r in rs)]
    if pairs:
        a, b = np.array(pairs).T
        report["serve_vs_eval_seed42"] = {"n": len(pairs), "mean_abs_psnr_diff": float(np.abs(a - b).mean()),
                                          "eval_mean": float(a.mean()), "serve_mean": float(b.mean())}

    headroom = {}
    for name in [*selectors, *METRICS]:
        tables = [np.array([r[name] for r in rs], dtype=float) for rs in groups.values()
                  if all(r.get(name) is not None for r in rs)]
        if not tables:
            continue
        within = float(np.mean([t.var() for t in tables]))
        total = float(np.concatenate(tables).var())
        headroom[name] = {"within_seed_std": float(np.sqrt(within)), "within_fraction": within / total if total else 0.0,
                          "best_minus_mean": float(np.mean([t.max() - t.mean() for t in tables]))}
    report["headroom"] = headroom

    correlation = {}
    for reward in selectors:
        correlation[reward] = {}
        for metric in METRICS:
            values = [spearman(np.array([r[reward] for r in rs], float), np.array([r[metric] for r in rs], float))
                      for rs in groups.values() if all(r.get(reward) is not None for r in rs)]
            values = [v for v in values if v is not None]
            correlation[reward][metric] = float(np.mean(values)) if values else None
    report["within_sample_spearman"] = correlation

    selection = {}
    for selector in [*selectors, *METRICS]:
        gains = {m: [] for m in METRICS}
        worst = {m: [] for m in METRICS}
        for rs in groups.values():
            if any(r.get(selector) is None for r in rs):
                continue
            values = np.array([r[selector] for r in rs], float)
            best, low = rs[int(values.argmax())], rs[int(values.argmin())]
            for metric in METRICS:
                mean = np.mean([r[metric] for r in rs])
                gains[metric].append(best[metric] - mean)
                worst[metric].append(low[metric] - mean)
        if gains[METRICS[0]]:
            selection[selector] = {
                "n": len(gains[METRICS[0]]),
                "best_vs_random": {m: bootstrap(np.array(v)) for m, v in gains.items()},
                "worst_vs_random": {m: float(np.mean(v)) for m, v in worst.items()},
            }
    report["best_of_n"] = selection
    report["oracle_note"] = "rows named after a metric select by that metric itself: an upper bound, not a reward"
    return report


def markdown(report: dict) -> str:
    def fmt(value, digits=3):
        return "—" if value is None else f"{value:.{digits}f}"

    lines = [f"# Phase 0: best-of-{len(report['seeds'])} by reward ({report['samples']} held-out samples)", ""]
    sanity = report.get("serve_vs_eval_seed42")
    if sanity:
        lines += [f"Seed 42 through the serve path vs the eval path: mean |ΔPSNR_late| = "
                  f"{sanity['mean_abs_psnr_diff']:.3f} dB over {sanity['n']} samples "
                  f"(eval {sanity['eval_mean']:.2f}, serve {sanity['serve_mean']:.2f}).", ""]
    lines += ["## Headroom across seeds", "", "| quantity | within-sample std | share of variance within sample "
              "| best seed − mean |", "|---|---|---|---|"]
    for name, row in report["headroom"].items():
        lines.append(f"| {name} | {fmt(row['within_seed_std'])} | {fmt(row['within_fraction'], 2)} "
                     f"| {fmt(row['best_minus_mean'])} |")
    lines += ["", "## Within-sample Spearman: reward vs independent metric", "",
              "| reward | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for reward, row in report["within_sample_spearman"].items():
        lines.append(f"| {reward} | " + " | ".join(fmt(row[m], 2) for m in METRICS) + " |")
    lines += ["", "## Best-of-N by selector: gain over a random seed (mean [95% CI])", "",
              "| selector | n | " + " | ".join(METRICS) + " | worst-of-N " + METRICS[0] + " |",
              "|---|---|" + "---|" * (len(METRICS) + 1)]
    for selector, row in report["best_of_n"].items():
        cells = [f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]" for m, lo, hi in (row["best_vs_random"][k] for k in METRICS)]
        tag = f"{selector} (oracle)" if selector in METRICS else selector
        lines.append(f"| {tag} | {row['n']} | " + " | ".join(cells) + f" | {row['worst_vs_random'][METRICS[0]]:+.3f} |")
    lines += ["", f"_{report['oracle_note']}_"]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("scores", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--json")
    p.add_argument("--seeds", help="comma-separated seeds to keep, e.g. to compare best-of-5 across models")
    args = p.parse_args()
    rows = [json.loads(line) for path in args.scores for line in Path(path).read_text().splitlines() if line.strip()]
    if args.seeds:
        keep = {int(s) for s in args.seeds.split(",")}
        rows = [row for row in rows if row["seed"] in keep]
    report = analyse(rows)
    Path(args.out).write_text(markdown(report), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
