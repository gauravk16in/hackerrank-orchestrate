"""
config.py — Central configuration, paths, and allowed-value enums.

Every other module imports from here. Keeping paths, provider settings, and the
controlled vocabularies (allowed values from problem_statement.md) in one place
makes the rest of the system small and consistent.

Secrets are read from environment variables only (never hardcoded), per AGENTS.md.
A local `code/.env` file is loaded automatically if present.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is a hard dep, but stay defensive
    load_dotenv = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent          # .../code
REPO_ROOT = CODE_DIR.parent                          # repo root
DATASET_DIR = REPO_ROOT / "dataset"                  # .../dataset
IMAGES_DIR = DATASET_DIR / "images"                  # .../dataset/images

# Input CSVs
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
CLAIMS_CSV = DATASET_DIR / "claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"

# Outputs
OUTPUT_CSV = DATASET_DIR / "output.csv"                       # predictions for claims.csv
EVAL_DIR = CODE_DIR / "evaluation"
SAMPLE_OUTPUT_CSV = EVAL_DIR / "sample_output.csv"           # predictions for sample_claims.csv
CACHE_DIR = CODE_DIR / ".cache"                              # VLM response cache

# Load .env (code/.env preferred, then repo-root .env) before reading env vars.
if load_dotenv is not None:
    for _candidate in (CODE_DIR / ".env", REPO_ROOT / ".env"):
        if _candidate.exists():
            load_dotenv(_candidate)
            break

# ---------------------------------------------------------------------------
# Provider / model configuration
# ---------------------------------------------------------------------------
# Pluggable: switch provider with the PROVIDER env var. Gemini is the primary
# implementation; the client layer in image_analyzer.py is provider-agnostic.
PROVIDER = os.getenv("PROVIDER", "gemini").strip().lower()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip()

# Amazon Bedrock (Claude vision via bedrock-runtime). Auth uses the standard AWS
# credential chain (env vars / profile / role) — no provider key needed here.
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1").strip()
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0").strip()

# Gemini accepts either GEMINI_API_KEY or GOOGLE_API_KEY.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def active_model() -> str:
    return {
        "gemini": GEMINI_MODEL,
        "openai": OPENAI_MODEL,
        "anthropic": ANTHROPIC_MODEL,
        "bedrock": BEDROCK_MODEL_ID,
    }.get(PROVIDER, GEMINI_MODEL)


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------
# Resize images whose encoded size exceeds this threshold (some test images are
# 5-8 MB, which wastes tokens/bandwidth and risks request-size limits).
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(4 * 1024 * 1024)))
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "1568"))  # longest edge after resize
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))

# ---------------------------------------------------------------------------
# Retry / rate-limit behavior
# ---------------------------------------------------------------------------
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "2.0"))   # seconds
RETRY_MAX_DELAY = float(os.getenv("RETRY_MAX_DELAY", "60.0"))
REQUEST_SLEEP = float(os.getenv("REQUEST_SLEEP", "5.0"))         # gentle throttle between calls

# ---------------------------------------------------------------------------
# Pricing assumptions (USD per 1M tokens) — used only for the cost report.
# These are configurable so the operational analysis can be re-derived if
# provider pricing changes. Documented in evaluation/evaluation_report.md.
# ---------------------------------------------------------------------------
PRICE_INPUT_PER_M = float(os.getenv("PRICE_INPUT_PER_M", "0.30"))   # gemini-2.5-flash text input
PRICE_OUTPUT_PER_M = float(os.getenv("PRICE_OUTPUT_PER_M", "2.50"))  # gemini-2.5-flash output
# Approx tokens charged per image (Gemini tiles large images ~258 tok/tile).
APPROX_TOKENS_PER_IMAGE = int(os.getenv("APPROX_TOKENS_PER_IMAGE", "560"))

# ---------------------------------------------------------------------------
# Allowed values (controlled vocabularies) — from problem_statement.md
# ---------------------------------------------------------------------------
CLAIM_OBJECTS = {"car", "laptop", "package"}

CLAIM_STATUSES = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
}

SEVERITIES = {"none", "low", "medium", "high", "unknown"}

# object_part allowed values are object-specific.
OBJECT_PARTS = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port",
        "base", "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label", "contents",
        "item", "unknown",
    },
}

# All object parts in one set (useful for loose validation / fuzzy matching).
ALL_OBJECT_PARTS = set().union(*OBJECT_PARTS.values())

RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}

# Canonical ordering for risk_flags in the output string. Derived from the
# ordering observed in sample_claims.csv: image-quality first, then mismatch,
# then authenticity, then history, then manual_review_required last.
RISK_FLAG_ORDER = [
    # quality
    "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    # mismatch / relevance
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    # authenticity
    "possible_manipulation", "non_original_image", "text_instruction_present",
    # history + review
    "user_history_risk", "manual_review_required",
]

# Flags that, when present, indicate a trust/mismatch concern strong enough to
# require human review (in addition to history-driven and contradiction-driven
# triggers). Derived from sample labels (e.g. case_002 wrong_object adds
# manual_review_required even with clean history; case_006 quality-only does not).
TRUST_MISMATCH_FLAGS = {
    "wrong_object", "wrong_object_part", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
}

# Flags that make the image set unusable for *automated* review (valid_image=false).
AUTHENTICITY_FLAGS = {"possible_manipulation", "non_original_image"}
