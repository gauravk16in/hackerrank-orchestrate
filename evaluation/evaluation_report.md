# Evaluation Report — Multi-Modal Evidence Review

Model: `gemini-2.5-flash` (provider: `gemini`). Scored on `dataset/sample_claims.csv` (3 labeled rows).

## 1. Strategy comparison

| strategy | claim_status acc | evidence_met acc | avg field acc | risk_flags F1 | calls | tokens(in/out) | est cost |
|---|---|---|---|---|---|---|---|
| `context` | 0.67 | 0.67 | 0.67 | 1.00 | 0 | 0/0 | $0.0 |
| `two_stage` | 0.67 | 0.67 | 0.67 | 0.80 | 0 | 0/0 | $0.0 |

**Final strategy chosen for `output.csv`: `context`** (highest claim_status accuracy: 0.67).

## 2. Per-field accuracy (best strategy)

| field | accuracy |
|---|---|
| evidence_standard_met | 0.67 |
| valid_image | 1.00 |
| issue_type | 0.67 |
| object_part | 1.00 |
| claim_status | 0.67 |
| severity | 0.00 |
| risk_flags (set F1) | 1.00 (P=1.00 R=1.00, exact=1.00) |
| supporting_image_ids (set F1) | 0.86 (P=1.00 R=0.75, exact=0.67) |

## 3. claim_status confusion matrix (best strategy)

| truth \ pred | supported | contradicted | not_enough_inf |
|---|---|---|---|
| supported | 2 | 0 | 0 |
| contradicted | 0 | 0 | 0 |
| not_enough_information | 1 | 0 | 0 |

## 4. Per-row results (best strategy)

| user_id | claim_status (pred/truth) | issue_type | object_part | severity | risk_flags exact |
|---|---|---|---|---|---|
| user_001 | supported/supported | dent/dent | rear_bumper/rear_bumper | high/medium ❌ | ✅ |
| user_002 | supported/not_enough_information ❌ | broken_part/broken_part | front_bumper/front_bumper | high/unknown ❌ | ✅ |
| user_004 | supported/supported | glass_shatter/crack ❌ | windshield/windshield | high/medium ❌ | ✅ |

## 5. Operational analysis

**Pricing assumptions:** `$0.3/1M` input tokens, `$2.5/1M` output tokens (typical gemini-2.5-flash tier; adjust constants in `config.py`). Large images are downscaled to a 1568px longest edge (~560 tokens/image).

### Sample run (measured)

- Model calls: **0** (cache hits: 3, errors: 0)
- Tokens: **0 in / 0 out** (~0 in / 0 out per call)
- Images processed: **0**
- Runtime: **0.6s** (~0.6s/claim)
- Measured sample cost: **$0.0**

### Full test-set projection

- Rows: **44**, images: **82**
- Model calls: **~0** (1 call/claim)
- Est tokens: **~0 in / 0 out**
- **Est cost to process full test set: ~$0.0**

### TPM/RPM, batching, caching, retries

- **1 model call per claim** (single multimodal call with all of that claim's images) — minimizes RPM and avoids redundant per-image calls.
- **On-disk response cache** (`code/.cache/`) keyed by model+prompt+image bytes: re-runs and duplicate inputs cost **zero** API calls. Duplicate `user_id`s across rows are still processed independently (correct), but identical (prompt, images) pairs hit cache.
- **Exponential backoff with jitter** (configurable `MAX_RETRIES`, base/cap delay) handles 429/5xx without hammering the API.
- **Optional throttle** (`REQUEST_SLEEP`) and small image footprint keep us well under typical free-tier TPM/RPM (e.g. Gemini Flash ~15 RPM / ~1M TPM). At ~1.7K input + ~0.5K output tokens/call and 44 calls, the full run is far below per-minute limits even without throttling.
- **Image downscaling** caps token/bandwidth cost from oversized (5–8 MB) images.
