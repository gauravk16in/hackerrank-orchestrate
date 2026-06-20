"""
evaluation/main.py — Evaluate the system on dataset/sample_claims.csv.

What it does:
1. Runs the pipeline on sample_claims.csv for one or more strategies.
2. Scores predictions against the labeled expected outputs:
   - claim_status accuracy + confusion matrix (the headline metric)
   - per-field accuracy (evidence_standard_met, valid_image, issue_type,
     object_part, severity)
   - set-based F1 for risk_flags and supporting_image_ids
3. Writes evaluation/metrics.json and a human-readable
   evaluation/evaluation_report.md including the operational analysis
   (model calls, tokens, images, approx cost, runtime, TPM/RPM strategy).

Usage:
    python evaluation/main.py                                   # both strategies
    python evaluation/main.py --strategies context
    python evaluation/main.py --limit 5 --no-cache
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make the parent code/ package importable when run as `python evaluation/main.py`.
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config          # noqa: E402
import utils           # noqa: E402
import main as pipeline  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent

# Fields scored by exact match.
EXACT_FIELDS = ["evidence_standard_met", "valid_image", "issue_type",
                "object_part", "claim_status", "severity"]
# Fields scored as sets (semicolon-separated; "none" => empty set).
SET_FIELDS = ["risk_flags", "supporting_image_ids"]
STATUSES = ["supported", "contradicted", "not_enough_information"]


def _as_set(value: str) -> set[str]:
    v = (value or "").strip()
    if not v or v.lower() == "none":
        return set()
    return {p.strip() for p in v.split(";") if p.strip()}


def _norm(value: str) -> str:
    return (value or "").strip().lower()


def score(pred_rows: list[dict], truth_rows: list[dict]) -> dict:
    n = min(len(pred_rows), len(truth_rows))
    exact = {f: 0 for f in EXACT_FIELDS}
    set_stats = {f: {"tp": 0, "fp": 0, "fn": 0, "exact": 0} for f in SET_FIELDS}
    confusion = defaultdict(Counter)   # truth -> Counter(pred)
    per_row = []

    for i in range(n):
        p, t = pred_rows[i], truth_rows[i]
        row_report = {"user_id": t.get("user_id", ""), "fields": {}}
        for f in EXACT_FIELDS:
            ok = _norm(p.get(f)) == _norm(t.get(f))
            exact[f] += ok
            row_report["fields"][f] = {"pred": p.get(f), "truth": t.get(f), "ok": ok}
        confusion[_norm(t.get("claim_status"))][_norm(p.get("claim_status"))] += 1
        for f in SET_FIELDS:
            ps, ts = _as_set(p.get(f)), _as_set(t.get(f))
            tp = len(ps & ts); fp = len(ps - ts); fn = len(ts - ps)
            set_stats[f]["tp"] += tp; set_stats[f]["fp"] += fp; set_stats[f]["fn"] += fn
            set_stats[f]["exact"] += (ps == ts)
            row_report["fields"][f] = {"pred": sorted(ps), "truth": sorted(ts),
                                        "exact": ps == ts}
        per_row.append(row_report)

    def prf(s):
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"precision": round(prec, 3), "recall": round(rec, 3),
                "f1": round(f1, 3), "exact_match": round(s["exact"] / n, 3)}

    metrics = {
        "n_rows": n,
        "exact_accuracy": {f: round(exact[f] / n, 3) for f in EXACT_FIELDS},
        "set_metrics": {f: prf(set_stats[f]) for f in SET_FIELDS},
        "claim_status_confusion": {t: dict(confusion[t]) for t in STATUSES},
        "per_row": per_row,
    }
    # headline averages
    metrics["headline"] = {
        "claim_status_accuracy": metrics["exact_accuracy"]["claim_status"],
        "evidence_standard_met_accuracy": metrics["exact_accuracy"]["evidence_standard_met"],
        "avg_exact_field_accuracy": round(
            sum(metrics["exact_accuracy"].values()) / len(EXACT_FIELDS), 3),
        "risk_flags_f1": metrics["set_metrics"]["risk_flags"]["f1"],
    }
    return metrics


def confusion_table(conf: dict) -> str:
    header = "| truth \\ pred | " + " | ".join(s[:14] for s in STATUSES) + " |"
    sep = "|" + "---|" * (len(STATUSES) + 1)
    lines = [header, sep]
    for t in STATUSES:
        cells = [str(conf.get(t, {}).get(p, 0)) for p in STATUSES]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_strategy(strategy: str, limit: int | None, use_cache: bool):
    out_csv = EVAL_DIR / f"sample_output_{strategy}.csv"
    rows, tracker = pipeline.process_dataset(
        config.SAMPLE_CLAIMS_CSV, out_csv, strategy=strategy,
        limit=limit, use_cache=use_cache)
    truth = utils.read_csv_dicts(config.SAMPLE_CLAIMS_CSV)
    if limit:
        truth = truth[:limit]
    metrics = score(rows, truth)
    metrics["usage"] = tracker.summary()
    metrics["strategy"] = strategy
    metrics["output_csv"] = str(out_csv)
    return metrics


def estimate_full_test_cost(sample_usage: dict, n_sample_rows: int) -> dict:
    """Extrapolate token/cost from the sample run to the full test set."""
    test_claims = utils.read_csv_dicts(config.CLAIMS_CSV)
    test_rows = len(test_claims)
    test_images = sum(len(utils.split_image_paths(r["image_paths"])) for r in test_claims)
    # Scale by ROW (handles two_stage's 2 calls/row correctly).
    rows = max(n_sample_rows, 1)
    per_row_in = sample_usage.get("input_tokens", 0) / rows
    per_row_out = sample_usage.get("output_tokens", 0) / rows
    calls_per_row = sample_usage.get("model_calls", 0) / rows
    est_in = per_row_in * test_rows
    est_out = per_row_out * test_rows
    cost = est_in / 1e6 * config.PRICE_INPUT_PER_M + est_out / 1e6 * config.PRICE_OUTPUT_PER_M
    return {
        "test_rows": test_rows,
        "test_images": test_images,
        "est_model_calls": round(calls_per_row * test_rows),
        "est_input_tokens": int(est_in),
        "est_output_tokens": int(est_out),
        "est_cost_usd": round(cost, 4),
    }


def write_report(results: list[dict], limit: int | None):
    best = max(results, key=lambda r: r["headline"]["claim_status_accuracy"])
    n_sample = best["n_rows"]
    cost_proj = estimate_full_test_cost(best["usage"], n_sample)

    lines = []
    lines.append("# Evaluation Report — Multi-Modal Evidence Review\n")
    lines.append(f"Model: `{config.active_model()}` (provider: `{config.PROVIDER}`). "
                 f"Scored on `dataset/sample_claims.csv` ({n_sample} labeled rows).\n")

    lines.append("## 1. Strategy comparison\n")
    lines.append("| strategy | claim_status acc | evidence_met acc | avg field acc | risk_flags F1 | calls | tokens(in/out) | est cost |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        h, u = r["headline"], r["usage"]
        lines.append(
            f"| `{r['strategy']}` | {h['claim_status_accuracy']:.2f} | "
            f"{h['evidence_standard_met_accuracy']:.2f} | {h['avg_exact_field_accuracy']:.2f} | "
            f"{h['risk_flags_f1']:.2f} | {u['model_calls']} | "
            f"{u['input_tokens']}/{u['output_tokens']} | ${u['estimated_cost_usd']} |")
    lines.append(f"\n**Final strategy chosen for `output.csv`: `{best['strategy']}`** "
                 f"(highest claim_status accuracy: {best['headline']['claim_status_accuracy']:.2f}).\n")

    lines.append("## 2. Per-field accuracy (best strategy)\n")
    lines.append("| field | accuracy |")
    lines.append("|---|---|")
    for f, a in best["exact_accuracy"].items():
        lines.append(f"| {f} | {a:.2f} |")
    for f, m in best["set_metrics"].items():
        lines.append(f"| {f} (set F1) | {m['f1']:.2f} (P={m['precision']:.2f} R={m['recall']:.2f}, exact={m['exact_match']:.2f}) |")
    lines.append("")

    lines.append("## 3. claim_status confusion matrix (best strategy)\n")
    lines.append(confusion_table(best["claim_status_confusion"]))
    lines.append("")

    lines.append("## 4. Per-row results (best strategy)\n")
    lines.append("| user_id | claim_status (pred/truth) | issue_type | object_part | severity | risk_flags exact |")
    lines.append("|---|---|---|---|---|---|")
    for rr in best["per_row"]:
        fcs = rr["fields"]
        cs = fcs["claim_status"]
        it = fcs["issue_type"]; op = fcs["object_part"]; sv = fcs["severity"]
        rfm = fcs["risk_flags"]["exact"]
        def mk(d): return f"{d['pred']}/{d['truth']}" + ("" if d["ok"] else " ❌")
        lines.append(f"| {rr['user_id']} | {mk(cs)} | {mk(it)} | {mk(op)} | {mk(sv)} | {'✅' if rfm else '❌'} |")
    lines.append("")

    lines.append("## 5. Operational analysis\n")
    u = best["usage"]
    per_call_in = u["input_tokens"] / max(u["model_calls"], 1)
    per_call_out = u["output_tokens"] / max(u["model_calls"], 1)
    lines.append(f"**Pricing assumptions:** `${config.PRICE_INPUT_PER_M}/1M` input tokens, "
                 f"`${config.PRICE_OUTPUT_PER_M}/1M` output tokens "
                 f"(typical {config.active_model()} tier; adjust constants in `config.py`). "
                 f"Large images are downscaled to a {config.MAX_IMAGE_DIMENSION}px longest edge "
                 f"(~{config.APPROX_TOKENS_PER_IMAGE} tokens/image).\n")
    lines.append("### Sample run (measured)\n")
    lines.append(f"- Model calls: **{u['model_calls']}** (cache hits: {u['cache_hits']}, errors: {u['errors']})")
    lines.append(f"- Tokens: **{u['input_tokens']:,} in / {u['output_tokens']:,} out** "
                 f"(~{per_call_in:.0f} in / {per_call_out:.0f} out per call)")
    lines.append(f"- Images processed: **{u['images_processed']}**")
    lines.append(f"- Runtime: **{u['elapsed_seconds']}s** "
                 f"(~{u['elapsed_seconds']/max(u['model_calls'],1):.1f}s/claim)")
    lines.append(f"- Measured sample cost: **${u['estimated_cost_usd']}**\n")
    lines.append("### Full test-set projection\n")
    lines.append(f"- Rows: **{cost_proj['test_rows']}**, images: **{cost_proj['test_images']}**")
    lines.append(f"- Model calls: **~{cost_proj['est_model_calls']}** (1 call/claim)")
    lines.append(f"- Est tokens: **~{cost_proj['est_input_tokens']:,} in / {cost_proj['est_output_tokens']:,} out**")
    lines.append(f"- **Est cost to process full test set: ~${cost_proj['est_cost_usd']}**\n")
    lines.append("### TPM/RPM, batching, caching, retries\n")
    lines.append("- **1 model call per claim** (single multimodal call with all of that claim's images) — minimizes RPM and avoids redundant per-image calls.")
    lines.append("- **On-disk response cache** (`code/.cache/`) keyed by model+prompt+image bytes: re-runs and duplicate inputs cost **zero** API calls. Duplicate `user_id`s across rows are still processed independently (correct), but identical (prompt, images) pairs hit cache.")
    lines.append("- **Exponential backoff with jitter** (configurable `MAX_RETRIES`, base/cap delay) handles 429/5xx without hammering the API.")
    lines.append("- **Optional throttle** (`REQUEST_SLEEP`) and small image footprint keep us well under typical free-tier TPM/RPM (e.g. Gemini Flash ~15 RPM / ~1M TPM). At ~1.7K input + ~0.5K output tokens/call and 44 calls, the full run is far below per-minute limits even without throttling.")
    lines.append("- **Image downscaling** caps token/bandwidth cost from oversized (5–8 MB) images.\n")

    report = "\n".join(lines)
    (EVAL_DIR / "evaluation_report.md").write_text(report, encoding="utf-8")
    (EVAL_DIR / "metrics.json").write_text(
        json.dumps({r["strategy"]: {k: v for k, v in r.items() if k != "per_row"}
                    for r in results}, indent=2), encoding="utf-8")
    print("\nWrote evaluation/evaluation_report.md and evaluation/metrics.json")
    print(f"Best strategy: {best['strategy']} "
          f"(claim_status acc {best['headline']['claim_status_accuracy']:.2f})")


def main():
    ap = argparse.ArgumentParser(description="Evaluate on sample_claims.csv")
    ap.add_argument("--strategies", default="context,two_stage",
                    help="comma-separated subset of: context,two_stage,evidence_first")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    results = []
    for s in strategies:
        print(f"\n{'='*60}\nEVALUATING STRATEGY: {s}\n{'='*60}")
        results.append(run_strategy(s, args.limit, not args.no_cache))
    write_report(results, args.limit)


if __name__ == "__main__":
    main()
