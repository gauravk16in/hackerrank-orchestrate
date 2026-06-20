"""
decision_engine.py — Turn raw VLM analysis into the final, validated output row.

Responsibilities (all deterministic):
1. Validate/snap every field onto its allowed value.
2. Merge image-grounded risk flags (from the VLM) with history flags (from the
   CSV) and derive `manual_review_required` from clear rules.
3. Apply consistency rules learned from sample_claims.csv (e.g. NEI -> severity
   unknown; issue_type none -> severity none; manipulation -> valid_image false).
4. Emit the 10 derived output columns in the exact required format.

History never *overrides* the image verdict — it only adds flags and review
requirements, consistent with the problem statement.
"""
from __future__ import annotations

import config
import utils
from claim_parser import ParsedClaim
from user_risk import UserRisk

# Output columns produced here (the 4 input columns are prepended by main.py).
DERIVED_COLUMNS = [
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]

FULL_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"] + DERIVED_COLUMNS


def _b(value, default=False) -> str:
    """Coerce to the literal 'true'/'false' strings used in the CSVs."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return "true" if value.strip().lower() in ("true", "yes", "1") else "false"
    return "true" if default else "false"


def _order_flags(flags: set[str]) -> list[str]:
    ordered = [f for f in config.RISK_FLAG_ORDER if f in flags]
    # any unexpected-but-valid flags not in the canonical order: append at end
    ordered += [f for f in flags if f in config.RISK_FLAGS and f not in ordered and f != "none"]
    return ordered


def make_decision(assessment: dict, parsed: ParsedClaim, risk: UserRisk,
                  claim_object: str, present_image_ids: list[str]) -> dict:
    claim_object = (claim_object or "").strip().lower()
    parts_allowed = config.OBJECT_PARTS.get(claim_object, config.ALL_OBJECT_PARTS)

    # --- enum validation ---------------------------------------------------
    claim_status = utils.closest_enum(
        assessment.get("claim_status"), config.CLAIM_STATUSES, "not_enough_information")
    issue_type = utils.closest_enum(
        assessment.get("visible_issue_type"), config.ISSUE_TYPES, "unknown")
    object_part = utils.closest_enum(
        assessment.get("object_part"), parts_allowed, "unknown")
    severity = utils.closest_enum(
        assessment.get("severity"), config.SEVERITIES, "unknown")

    evidence_met = bool(assessment.get("evidence_standard_met", False))
    valid_image = bool(assessment.get("valid_image", True))

    # --- risk flags --------------------------------------------------------
    flags: set[str] = set()
    for f in assessment.get("risk_flags", []) or []:
        snapped = utils.closest_enum(f, config.RISK_FLAGS, "none")
        if snapped != "none":
            flags.add(snapped)
    # drop history/review flags if the model emitted them — we derive these.
    flags.discard("user_history_risk")
    flags.discard("manual_review_required")

    # authenticity from explicit booleans
    if assessment.get("manipulation_suspected"):
        # keep whichever the model already chose; default to possible_manipulation
        if not (flags & config.AUTHENTICITY_FLAGS):
            flags.add("possible_manipulation")
    # text-instruction present: from the image OR the conversation (defense in depth)
    if assessment.get("text_instruction_present") or parsed.adversarial:
        flags.add("text_instruction_present")

    # authenticity -> not valid for automated review
    if flags & config.AUTHENTICITY_FLAGS:
        valid_image = False

    # history-derived flags (deterministic, from CSV)
    history_flags = set(risk.history_flags)
    if "user_history_risk" in history_flags:
        flags.add("user_history_risk")

    # manual_review_required rule:
    #  - present in history, OR
    #  - the claim is contradicted, OR
    #  - a trust/mismatch flag is present (wrong_object, claim_mismatch, manipulation,
    #    non_original, text_instruction_present, wrong_object_part)
    needs_review = (
        "manual_review_required" in history_flags
        or claim_status == "contradicted"
        or bool(flags & config.TRUST_MISMATCH_FLAGS)
    )
    if needs_review:
        flags.add("manual_review_required")

    ordered_flags = _order_flags(flags)
    risk_flags_str = ";".join(ordered_flags) if ordered_flags else "none"

    # --- supporting image ids ---------------------------------------------
    raw_support = assessment.get("supporting_image_ids", []) or []
    present = list(present_image_ids)
    support = [s for s in raw_support if s in present]
    # safety: a supported claim must cite at least one usable image if any exist
    if claim_status == "supported" and not support and present:
        usable = [im.get("image_id") for im in assessment.get("images", [])
                  if im.get("usable") and im.get("image_id") in present]
        support = usable or present[:1]
    support_str = ";".join(support) if support else "none"

    # --- consistency rules (learned from sample labels) --------------------
    if claim_status == "not_enough_information":
        severity = "unknown"
        evidence_met = False
    elif issue_type == "none":
        severity = "none"

    # --- justifications ----------------------------------------------------
    evidence_reason = (assessment.get("evidence_standard_met_reason") or "").strip() \
        or ("The image set is sufficient to evaluate the claim." if evidence_met
            else "The image set is not sufficient to evaluate the claimed condition.")
    justification = (assessment.get("claim_status_justification") or "").strip() \
        or "Decision based on the submitted image evidence."
    # Append a concise history note when history adds risk and it's not mentioned.
    if risk.has_risk and "histor" not in justification.lower():
        note = risk.history_summary.strip().rstrip(".")
        if note:
            justification = f"{justification} User history adds risk context: {note}."

    return {
        "evidence_standard_met": _b(evidence_met),
        "evidence_standard_met_reason": evidence_reason,
        "risk_flags": risk_flags_str,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": justification,
        "supporting_image_ids": support_str,
        "valid_image": _b(valid_image),
        "severity": severity,
    }
