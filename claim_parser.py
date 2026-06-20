"""
claim_parser.py — Extract what the user is actually claiming from the chat.

The conversation defines *what to check*. We parse it deterministically (no LLM
call) because the transcript structure is consistent (`Customer:` / `Support:` /
`Agent:` turns) and damage vocabulary is finite. The result is a structured
*hint* — the VLM still receives the raw customer statements and makes the final
call grounded in the images. The parser also performs a first line of defense
against prompt-injection by flagging adversarial instructions in the text.

Conversations may be in English, Hindi, or Hinglish; damage/part nouns are
frequently written in English even in Hinglish text, so keyword matching still
provides useful signal. When it fails, the VLM falls back to the raw transcript.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field

import config

# --- keyword tables ---------------------------------------------------------
# issue_type keywords (order matters: more specific phrases first).
ISSUE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("glass_shatter", ["shattered glass", "glass shattered", "shatter", "smashed glass"]),
    ("crack", ["crack", "cracked", "cracking", "hairline", "spider", "spreading"]),
    ("dent", ["dent", "dented", "dinged", "ding", "deform", "caved", "bent in"]),
    ("scratch", ["scratch", "scrape", "scraped", "scuff", "scuffed", "mark", "paint mark"]),
    ("crushed_packaging", ["crushed", "crush", "smashed box", "caved in", "compressed"]),
    ("torn_packaging", ["torn", "tear", "ripped", "torn open", "tape broken", "phati"]),
    ("water_damage", ["water damage", "water-damaged", "wet", "soaked", "moisture", "liquid", "spill"]),
    ("stain", ["stain", "stained", "discolor", "patch", "sticky"]),
    ("missing_part", ["missing", "not inside", "not there", "empty", "gone", "absent", "no item", "without the"]),
    ("broken_part", ["broken", "broke", "break", "snapped", "not sitting", "fell off", "detached", "shattered", "damaged part"]),
]

# part keywords per object (more specific phrases first).
PART_KEYWORDS: dict[str, list[tuple[str, list[str]]]] = {
    "car": [
        ("front_bumper", ["front bumper", "front-bumper", "bumper front", "front side bumper"]),
        ("rear_bumper", ["rear bumper", "back bumper", "rear-bumper", "bumper at the back", "back of the car", "rear side"]),
        ("windshield", ["windshield", "windscreen", "front glass", "front windshield"]),
        ("side_mirror", ["side mirror", "wing mirror", "rear-view mirror", "side-mirror", "mirror"]),
        ("headlight", ["headlight", "head light", "head lamp", "front light", "left light", "right light"]),
        ("taillight", ["taillight", "tail light", "rear light", "back light", "brake light"]),
        ("hood", ["hood", "bonnet", "top panel", "front top panel"]),
        ("door", ["door panel", "door", "side door"]),
        ("fender", ["fender", "wheel arch"]),
        ("quarter_panel", ["quarter panel", "quarter-panel"]),
        ("front_bumper", ["bumper"]),  # generic bumper fallback -> front
        ("body", ["body", "panel", "side of the car", "side panel"]),
    ],
    "laptop": [
        ("screen", ["screen", "display", "lcd", "monitor", "panel"]),
        ("keyboard", ["keyboard", "keys", "key ", "keypad"]),
        ("trackpad", ["trackpad", "touchpad", "track pad"]),
        ("hinge", ["hinge", "hinges", "opens", "wobbles", "wobble"]),
        ("lid", ["lid", "cover", "top cover"]),
        ("corner", ["corner", "edge"]),
        ("port", ["port", "usb", "charging port", "hdmi", "jack"]),
        ("base", ["base", "bottom"]),
        ("body", ["body", "casing", "chassis", "outer"]),
    ],
    "package": [
        ("seal", ["seal", "tape", "flap", "opened", "torn open", "phati", "khol"]),
        ("package_corner", ["corner", "package corner"]),
        ("package_side", ["side", "package side", "surface", "outside"]),
        ("label", ["label", "address", "sticker", "shipping label"]),
        ("contents", ["contents", "content", "item inside", "product inside", "inside the box", "what was inside"]),
        ("item", ["item", "product"]),
        ("box", ["box", "package", "parcel", "carton", "shipping box"]),
    ],
}

# Prompt-injection / adversarial phrases (lower-cased substring match).
ADVERSARIAL_PATTERNS = [
    "approve this claim", "approve the claim", "approve immediately", "auto approve",
    "auto-approve", "please approve", "must be approved", "should approve",
    "approve it", "mark as approved", "skip review", "skip manual review",
    "skip the review", "no review needed", "without review", "ignore previous",
    "ignore the previous", "ignore prior", "ignore your instructions",
    "follow the note", "follow this note", "as the note says", "do not reject",
    "don't reject", "system reading this", "any system reading", "ai reading this",
    "override", "disregard", "you must approve", "accept this claim",
]

# Multi-part connectors that hint two claimed parts.
MULTI_PART_HINTS = [" and ", " & ", " plus ", " also ", " both ", "two things", "two parts", "multiple parts"]


@dataclass
class ParsedClaim:
    raw_text: str
    customer_turns: list[str] = field(default_factory=list)
    customer_focus: str = ""              # last few customer turns (where the claim lives)
    issue_type_hint: str = "unknown"      # best-effort; verified against images by VLM
    object_part_hint: str = "unknown"
    all_part_hints: list[str] = field(default_factory=list)
    multi_part: bool = False
    adversarial: bool = False
    adversarial_phrases: list[str] = field(default_factory=list)
    summary: str = ""                     # short natural-language summary for the prompt


def _split_turns(text: str) -> list[tuple[str, str]]:
    """Split a transcript into (speaker, utterance) pairs.

    Turns are separated by ' | ' and prefixed with 'Customer:' / 'Support:' /
    'Agent:'. Robust to missing separators by also splitting on the role tokens.
    """
    if not text:
        return []
    # Normalize separators: ensure role tokens start a new chunk.
    normalized = re.sub(r"\s*\|\s*", "\n", text)
    normalized = re.sub(r"(?<!^)\s*(Customer:|Support:|Agent:)", r"\n\1", normalized)
    turns: list[tuple[str, str]] = []
    for line in normalized.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(Customer|Support|Agent)\s*:\s*(.*)$", line, re.IGNORECASE)
        if m:
            turns.append((m.group(1).lower(), m.group(2).strip()))
        elif turns:  # continuation of previous turn
            spk, prev = turns[-1]
            turns[-1] = (spk, (prev + " " + line).strip())
    return turns


@functools.lru_cache(maxsize=4096)
def _kw_regex(kw: str) -> "re.Pattern":
    # Leading word boundary only: matches "dent" in "dented" (stemming) but NOT
    # "side" inside "inside" or "port" inside "report".
    return re.compile(r"\b" + re.escape(kw), re.IGNORECASE)


def _match_first(text_lc: str, table: list[tuple[str, list[str]]]) -> tuple[str, list[str]]:
    """Return (first matching label, all matching labels in order) from a table."""
    matches: list[str] = []
    for label, kws in table:
        for kw in kws:
            if _kw_regex(kw).search(text_lc):
                if label not in matches:
                    matches.append(label)
                break
    first = matches[0] if matches else "unknown"
    return first, matches


def parse_claim(user_claim: str, claim_object: str) -> ParsedClaim:
    claim_object = (claim_object or "").strip().lower()
    turns = _split_turns(user_claim or "")
    customer_turns = [u for spk, u in turns if spk == "customer"]
    # The actual claim almost always lives in the last 1-3 customer turns.
    focus_turns = customer_turns[-3:] if customer_turns else []
    focus = " ".join(focus_turns)
    # Use full customer text for issue/part detection (earlier turns add context).
    all_customer = " ".join(customer_turns) if customer_turns else (user_claim or "")
    text_lc = all_customer.lower()
    focus_lc = focus.lower()

    issue_type_hint, _ = _match_first(focus_lc or text_lc, ISSUE_KEYWORDS)
    if issue_type_hint == "unknown":
        issue_type_hint, _ = _match_first(text_lc, ISSUE_KEYWORDS)

    part_table = PART_KEYWORDS.get(claim_object, [])
    object_part_hint, all_parts = _match_first(focus_lc or text_lc, part_table)
    if object_part_hint == "unknown":
        object_part_hint, all_parts = _match_first(text_lc, part_table)

    # Multi-part detection.
    multi_part = (
        len(all_parts) >= 2
        or (any(h in text_lc for h in MULTI_PART_HINTS) and len(all_parts) >= 1)
    )

    # Adversarial detection across the WHOLE transcript (injection can appear in
    # any turn, including text the customer claims to be quoting).
    full_lc = (user_claim or "").lower()
    adversarial_phrases = [p for p in ADVERSARIAL_PATTERNS if p in full_lc]
    adversarial = len(adversarial_phrases) > 0

    summary = _build_summary(claim_object, issue_type_hint, object_part_hint, all_parts, focus)

    return ParsedClaim(
        raw_text=user_claim or "",
        customer_turns=customer_turns,
        customer_focus=focus,
        issue_type_hint=issue_type_hint,
        object_part_hint=object_part_hint,
        all_part_hints=all_parts,
        multi_part=multi_part,
        adversarial=adversarial,
        adversarial_phrases=adversarial_phrases,
        summary=summary,
    )


def _build_summary(claim_object: str, issue: str, part: str,
                   all_parts: list[str], focus: str) -> str:
    parts_txt = " and ".join(all_parts) if len(all_parts) >= 2 else (part if part != "unknown" else "unspecified part")
    issue_txt = issue if issue != "unknown" else "an unspecified issue"
    return f"User reports {issue_txt} on the {claim_object} {parts_txt}."
