"""
evidence_checker.py — Select the minimum image-evidence requirements that apply
to a claim, so the VLM knows the bar the images must clear.

We load evidence_requirements.csv and map (claim_object, issue_type, #images)
onto the relevant requirement rows. The requirement text is injected into the
VLM prompt as the evidence standard; the model decides whether the submitted
images meet it (`evidence_standard_met`).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import config
import utils

# Map an issue_type to the object-specific requirement IDs it triggers.
# (General + trust requirements are always included.)
ISSUE_TO_REQUIREMENTS: dict[str, dict[str, list[str]]] = {
    "car": {
        "dent": ["REQ_CAR_BODY_PANEL"],
        "scratch": ["REQ_CAR_BODY_PANEL"],
        "crack": ["REQ_CAR_GLASS_LIGHT_MIRROR"],
        "glass_shatter": ["REQ_CAR_GLASS_LIGHT_MIRROR"],
        "broken_part": ["REQ_CAR_GLASS_LIGHT_MIRROR"],
        "missing_part": ["REQ_CAR_GLASS_LIGHT_MIRROR"],
    },
    "laptop": {
        "crack": ["REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD"],
        "glass_shatter": ["REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD"],
        "stain": ["REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD"],
        "water_damage": ["REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD"],
        "missing_part": ["REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD"],
        "dent": ["REQ_LAPTOP_BODY_HINGE_PORT"],
        "broken_part": ["REQ_LAPTOP_BODY_HINGE_PORT", "REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD"],
        "scratch": ["REQ_LAPTOP_BODY_HINGE_PORT"],
    },
    "package": {
        "crushed_packaging": ["REQ_PACKAGE_EXTERIOR"],
        "torn_packaging": ["REQ_PACKAGE_EXTERIOR"],
        "broken_part": ["REQ_PACKAGE_EXTERIOR"],
        "water_damage": ["REQ_PACKAGE_LABEL_OR_STAIN"],
        "stain": ["REQ_PACKAGE_LABEL_OR_STAIN"],
        "missing_part": ["REQ_PACKAGE_CONTENTS"],
    },
}

# Part-based hints when issue_type is unknown (e.g. claim mentions a part only).
PART_TO_REQUIREMENTS: dict[str, dict[str, list[str]]] = {
    "car": {
        "windshield": ["REQ_CAR_GLASS_LIGHT_MIRROR"],
        "headlight": ["REQ_CAR_GLASS_LIGHT_MIRROR"],
        "taillight": ["REQ_CAR_GLASS_LIGHT_MIRROR"],
        "side_mirror": ["REQ_CAR_GLASS_LIGHT_MIRROR"],
    },
    "package": {
        "contents": ["REQ_PACKAGE_CONTENTS"],
        "item": ["REQ_PACKAGE_CONTENTS"],
        "label": ["REQ_PACKAGE_LABEL_OR_STAIN"],
        "seal": ["REQ_PACKAGE_EXTERIOR"],
    },
}

ALWAYS_INCLUDE = ["REQ_GENERAL_OBJECT_PART", "REQ_REVIEW_TRUST"]


@dataclass
class EvidenceRequirement:
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str


@lru_cache(maxsize=1)
def _requirements_index() -> dict[str, EvidenceRequirement]:
    rows = utils.read_csv_dicts(config.EVIDENCE_REQUIREMENTS_CSV)
    out: dict[str, EvidenceRequirement] = {}
    for r in rows:
        out[r["requirement_id"].strip()] = EvidenceRequirement(
            requirement_id=r["requirement_id"].strip(),
            claim_object=r["claim_object"].strip(),
            applies_to=r["applies_to"].strip(),
            minimum_image_evidence=r["minimum_image_evidence"].strip(),
        )
    return out


def get_requirements(claim_object: str, issue_type_hint: str,
                     object_part_hint: str = "unknown",
                     num_images: int = 1) -> list[EvidenceRequirement]:
    claim_object = (claim_object or "").strip().lower()
    issue = (issue_type_hint or "unknown").strip().lower()
    part = (object_part_hint or "unknown").strip().lower()
    index = _requirements_index()

    ids: list[str] = list(ALWAYS_INCLUDE)
    if num_images > 1:
        ids.append("REQ_GENERAL_MULTI_IMAGE")

    ids += ISSUE_TO_REQUIREMENTS.get(claim_object, {}).get(issue, [])
    if part != "unknown":
        ids += PART_TO_REQUIREMENTS.get(claim_object, {}).get(part, [])

    # If we still have no object-specific requirement, include the broad
    # exterior/body requirement for that object as a sensible default.
    object_defaults = {
        "car": "REQ_CAR_BODY_PANEL",
        "laptop": "REQ_LAPTOP_BODY_HINGE_PORT",
        "package": "REQ_PACKAGE_EXTERIOR",
    }
    if not any(i not in ALWAYS_INCLUDE and i != "REQ_GENERAL_MULTI_IMAGE" for i in ids):
        default_id = object_defaults.get(claim_object)
        if default_id:
            ids.append(default_id)

    # De-duplicate preserving order, resolve to requirement objects.
    seen: set[str] = set()
    result: list[EvidenceRequirement] = []
    for rid in ids:
        if rid in seen:
            continue
        seen.add(rid)
        req = index.get(rid)
        if req:
            result.append(req)
    return result


def requirements_text(reqs: list[EvidenceRequirement]) -> str:
    """Render requirements as a compact bullet list for the prompt."""
    return "\n".join(f"- [{r.requirement_id}] {r.minimum_image_evidence}" for r in reqs)
