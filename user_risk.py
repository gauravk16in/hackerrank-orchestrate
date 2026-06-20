"""
user_risk.py — Look up historical risk context for a user.

User history adds *risk context only*. Per the problem statement, it must never
override clear visual evidence on its own. The history_flags column is already
pre-computed and semicolon-formatted (`none`, `user_history_risk`,
`user_history_risk;manual_review_required`, `manual_review_required`), so we use
those directly and merge them deterministically into the final risk_flags. The
history_summary provides justification text for the prompt and output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import config
import utils


@dataclass
class UserRisk:
    user_id: str
    found: bool = False
    past_claim_count: int = 0
    accepted: int = 0
    manual_review: int = 0
    rejected: int = 0
    last_90_days: int = 0
    history_flags: list[str] = field(default_factory=list)   # excludes "none"
    history_summary: str = ""
    rejection_rate: float = 0.0

    @property
    def has_risk(self) -> bool:
        return bool(self.history_flags)

    def context_for_prompt(self) -> str:
        if not self.found:
            return "No prior history on record for this user."
        flags = ", ".join(self.history_flags) if self.history_flags else "none"
        return (
            f"Past claims: {self.past_claim_count} "
            f"(accepted={self.accepted}, manual_review={self.manual_review}, "
            f"rejected={self.rejected}, last_90_days={self.last_90_days}). "
            f"Rejection rate: {self.rejection_rate:.0%}. "
            f"History flags: {flags}. "
            f"Summary: {self.history_summary}"
        )


def _to_int(v) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return 0


@lru_cache(maxsize=1)
def _history_index() -> dict[str, dict]:
    rows = utils.read_csv_dicts(config.USER_HISTORY_CSV)
    return {r["user_id"].strip(): r for r in rows}


def assess_risk(user_id: str) -> UserRisk:
    user_id = (user_id or "").strip()
    row = _history_index().get(user_id)
    if not row:
        return UserRisk(user_id=user_id, found=False)

    past = _to_int(row.get("past_claim_count"))
    rejected = _to_int(row.get("rejected_claim"))
    raw_flags = (row.get("history_flags") or "").strip()
    flags = [f.strip() for f in raw_flags.split(";") if f.strip() and f.strip() != "none"]

    return UserRisk(
        user_id=user_id,
        found=True,
        past_claim_count=past,
        accepted=_to_int(row.get("accept_claim")),
        manual_review=_to_int(row.get("manual_review_claim")),
        rejected=rejected,
        last_90_days=_to_int(row.get("last_90_days_claim_count")),
        history_flags=flags,
        history_summary=(row.get("history_summary") or "").strip(),
        rejection_rate=(rejected / past) if past else 0.0,
    )
