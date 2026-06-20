"""
main.py — Orchestrator / terminal entry point.

Wires the pipeline together for each claim row:

    claim row
      -> claim_parser.parse_claim         (what is being claimed + injection check)
      -> user_risk.assess_risk            (history risk context)
      -> evidence_checker.get_requirements(the evidence bar)
      -> load + downscale images
      -> image_analyzer.analyze_claim     (the VLM call; cached)
      -> decision_engine.make_decision    (validate + merge + finalize)
      -> output row

Usage:
    python main.py --mode sample          # -> code/evaluation/sample_output.csv
    python main.py --mode test            # -> dataset/output.csv
    python main.py --mode test --limit 5  # quick smoke test on first 5 rows
    python main.py --mode sample --strategy evidence_first --no-cache

Each ROW is an independent claim. We never deduplicate by user_id (claims.csv
intentionally contains repeated user_ids), so the output always has exactly one
row per input row.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import config
import utils
import claim_parser
import user_risk
import evidence_checker as ev
import image_analyzer
import decision_engine as de
from decision_engine import FULL_COLUMNS
from utils import log


def _customer_statements(parsed: claim_parser.ParsedClaim) -> str:
    if parsed.customer_turns:
        return "\n".join(f"- {t}" for t in parsed.customer_turns)
    return parsed.raw_text or "(no conversation provided)"


def process_row(row: dict, client, tracker: utils.CostTracker,
                strategy: str = "context", use_cache: bool = True) -> dict:
    """Process a single claim row -> full output row (14 columns)."""
    user_id = row.get("user_id", "").strip()
    image_paths = row.get("image_paths", "").strip()
    user_claim = row.get("user_claim", "")
    claim_object = row.get("claim_object", "").strip().lower()

    parsed = claim_parser.parse_claim(user_claim, claim_object)
    risk = user_risk.assess_risk(user_id)
    rel_paths = utils.split_image_paths(image_paths)
    images = [utils.load_image(p) for p in rel_paths]
    present_ids = [im.image_id for im in images if im.exists]
    reqs = ev.get_requirements(claim_object, parsed.issue_type_hint,
                               parsed.object_part_hint, len(rel_paths))

    # No usable images at all -> deterministic not_enough_information (no API call).
    if not any(im.exists and im.data for im in images):
        log.warning("  no usable images for %s (%s)", user_id, image_paths)
        derived = de.make_decision(
            {"claim_status": "not_enough_information", "evidence_standard_met": False,
             "valid_image": False, "visible_issue_type": "unknown",
             "object_part": "unknown", "severity": "unknown",
             "supporting_image_ids": [], "risk_flags": [],
             "evidence_standard_met_reason": "No usable images were provided.",
             "claim_status_justification": "No usable image evidence was available to evaluate the claim."},
            parsed, risk, claim_object, present_ids)
    else:
        result = image_analyzer.analyze_claim(
            client,
            claim_object=claim_object,
            parsed_summary=parsed.summary,
            customer_statements=_customer_statements(parsed),
            requirements_text=ev.requirements_text(reqs),
            history_context=risk.context_for_prompt(),
            images=images,
            adversarial=parsed.adversarial,
            adversarial_phrases=parsed.adversarial_phrases,
            strategy=strategy,
            use_cache=use_cache,
        )
        if result.cached:
            tracker.record(cached=True)
        elif result.error:
            tracker.record_error()
        else:
            tracker.record(input_tokens=result.input_tokens,
                           output_tokens=result.output_tokens,
                           images=result.images_sent, calls=result.n_calls)
        derived = de.make_decision(result.assessment, parsed, risk,
                                   claim_object, present_ids)

    out = {"user_id": user_id, "image_paths": image_paths,
           "user_claim": user_claim, "claim_object": claim_object}
    out.update(derived)
    return out


def _row_key(row: dict) -> tuple:
    """Stable identity for a claim row (works on both input and output rows)."""
    return (
        row.get("user_id", "").strip(),
        row.get("image_paths", "").strip(),
        row.get("user_claim", ""),
        row.get("claim_object", "").strip().lower(),
    )


def _merge_and_write(all_rows: list[dict], output_csv, processed: dict) -> list[dict]:
    """Merge this batch's results with any rows already in output_csv (from prior
    batches), then write the full file in original input order. This lets you run
    the dataset in offset/limit batches without overwriting earlier results."""
    existing: dict[tuple, dict] = {}
    if Path(output_csv).exists():
        try:
            for r in utils.read_csv_dicts(output_csv):
                existing[_row_key(r)] = r
        except Exception:
            pass
    existing.update(processed)  # newly processed rows win over stale ones
    final = [existing[_row_key(r)] for r in all_rows if _row_key(r) in existing]
    utils.write_csv_dicts(output_csv, final, FULL_COLUMNS)
    return final


def process_dataset(input_csv, output_csv, strategy: str = "context",
                    limit: int | None = None, offset: int = 0,
                    use_cache: bool = True) -> tuple[list[dict], utils.CostTracker]:
    all_rows = utils.read_csv_dicts(input_csv)
    n_all = len(all_rows)
    start = max(0, offset)
    end = n_all if limit is None else min(n_all, start + limit)
    batch = all_rows[start:end]

    tracker = utils.CostTracker()
    client = None  # lazily created so import/no-image paths don't require a key
    processed: dict[tuple, dict] = {}

    log.info("Processing rows %d-%d of %d from %s (strategy=%s, cache=%s)",
             start + 1, end, n_all, input_csv, strategy, use_cache)
    for bi, row in enumerate(batch, 1):
        gidx = start + bi              # global 1-based row number
        uid = row.get("user_id", "?")
        log.info("Processing claim %d/%d (batch %d/%d) for %s ...",
                 gidx, n_all, bi, len(batch), uid)
        try:
            if client is None:
                client = image_analyzer.get_client()
            processed[_row_key(row)] = process_row(row, client, tracker, strategy, use_cache)

            time.sleep(5)
        except Exception as exc:  # never let one row kill the run
            log.error("  row %d (%s) failed hard: %s", gidx, uid, exc)
            tracker.record_error()
            processed[_row_key(row)] = _fallback_row(row, str(exc))

    out_rows = _merge_and_write(all_rows, output_csv, processed)
    log.info("Wrote %d/%d rows -> %s (this batch processed %d)",
             len(out_rows), n_all, output_csv, len(batch))
    if len(out_rows) < n_all:
        log.info("  %d row(s) not yet processed — run remaining offsets to complete the file.",
                 n_all - len(out_rows))
    log.info("USAGE: %s", tracker.pretty())
    return out_rows, tracker


def _fallback_row(row: dict, error: str) -> dict:
    return {
        "user_id": row.get("user_id", ""),
        "image_paths": row.get("image_paths", ""),
        "user_claim": row.get("user_claim", ""),
        "claim_object": row.get("claim_object", "").strip().lower(),
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": "Processing error; could not evaluate evidence.",
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": f"Automated processing failed ({error}); routed to manual review.",
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown",
    }


def main():
    ap = argparse.ArgumentParser(description="Multi-modal damage-claim evidence review")
    ap.add_argument("--mode", choices=["sample", "test"], default="test",
                    help="sample = sample_claims.csv, test = claims.csv")
    ap.add_argument("--strategy", choices=["context", "evidence_first", "two_stage"],
                    default="context")
    ap.add_argument("--limit", type=int, default=None, help="process only N rows (from --offset)")
    ap.add_argument("--offset", "--start", type=int, default=0, dest="offset",
                    help="skip the first N rows; combine with --limit for batches")
    ap.add_argument("--no-cache", action="store_true", help="bypass the response cache")
    ap.add_argument("--output", default=None, help="override output CSV path")
    args = ap.parse_args()

    if args.mode == "sample":
        input_csv = config.SAMPLE_CLAIMS_CSV
        output_csv = args.output or config.SAMPLE_OUTPUT_CSV
    else:
        input_csv = config.CLAIMS_CSV
        output_csv = args.output or config.OUTPUT_CSV

    t0 = time.time()
    _, tracker = process_dataset(input_csv, output_csv, strategy=args.strategy,
                                 limit=args.limit, offset=args.offset,
                                 use_cache=not args.no_cache)
    s = tracker.summary()
    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    for k, v in s.items():
        print(f"  {k:22}: {v}")
    print(f"  wall_clock_seconds    : {round(time.time() - t0, 1)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
