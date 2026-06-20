# Multi-Modal Evidence Review

A system that verifies damage claims (car / laptop / package) by reasoning over
**submitted images**, a **claim conversation**, **user history**, and
**minimum evidence requirements**. For each claim it decides whether the images
**support**, **contradict**, or provide **not_enough_information** for the claim,
and emits structured fields (issue type, object part, severity, risk flags,
supporting image ids, evidence sufficiency).

The **images are the primary source of truth.** The conversation defines *what
to check*; user history only adds *risk context* and never overrides clear
visual evidence. The system is hardened against **prompt injection** in both the
conversation and inside the images themselves.

---

## Architecture

```
claims.csv row
  ├─ claim_parser.py     → extract the claim + detect adversarial text (no LLM)
  ├─ user_risk.py        → look up history flags / summary (deterministic)
  ├─ evidence_checker.py → select applicable minimum-evidence requirements
  ├─ image_analyzer.py   → ONE multimodal VLM call (cached) → structured JSON
  │     └─ prompts.py    → system + user prompt (security rules, enums, schema)
  └─ decision_engine.py  → validate enums, merge flags, apply rules → output row
```

| File | Responsibility |
|---|---|
| `config.py` | Paths, provider/model config, allowed-value enums, pricing constants |
| `utils.py` | Image load/downscale, ID/path resolution, retry, cost tracking, CSV I/O, fuzzy enum matching |
| `claim_parser.py` | Parse the transcript; best-effort issue/part hints; prompt-injection detection |
| `user_risk.py` | `user_history.csv` lookup → flags + summary (risk context only) |
| `evidence_checker.py` | Map (object, issue) → minimum image-evidence requirements |
| `prompts.py` | The VLM prompt: role, security rules, rubric, allowed values, JSON schema |
| `image_analyzer.py` | Pluggable VLM client (Gemini primary), structured output, caching, retries, tokens |
| `decision_engine.py` | Deterministic validation + flag merging + consistency rules → final 14 columns |
| `main.py` | Orchestrator / CLI entry point |
| `evaluation/main.py` | Metrics, strategy comparison, `evaluation_report.md` |

**Design choice — deterministic where it matters.** Only the *visual judgment*
uses the model; claim parsing, history risk, requirement selection, enum
validation, flag merging, and the manual-review rule are all deterministic code.
This keeps results reproducible and the model focused on what it's good at.

---

## Setup

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then put your key in .env
# .env:  PROVIDER=gemini   GEMINI_API_KEY=...   (or GOOGLE_API_KEY)
```

Secrets are read from environment variables only (never hardcoded). The provider
is pluggable via `PROVIDER` (`gemini` is the implemented primary; `openai` /
`anthropic` are stubs ready to wire up).

---

## Running

```bash
# Smoke test on the first few rows (cheap):
python main.py --mode test --limit 3

# Full predictions for dataset/claims.csv -> dataset/output.csv
python main.py --mode test

# Run on the labeled sample set -> code/evaluation/sample_output.csv
python main.py --mode sample

# Evaluate + compare strategies + (re)generate the report:
python evaluation/main.py
```

### Batched runs (rate-limit friendly)

`--offset` + `--limit` process the dataset in chunks. Batches **accumulate**
into the same `output.csv` (later batches merge with earlier ones; they do not
overwrite), so you can work around per-minute / quota limits:

```bash
python main.py --mode test --limit 5 --offset 0    # rows 1-5
python main.py --mode test --limit 5 --offset 5    # rows 6-10  (file now has 10)
python main.py --mode test --limit 5 --offset 10   # rows 11-15 (file now has 15)
# ... continue until all 44 rows are present
```

Flags: `--strategy {context,two_stage,evidence_first}`, `--limit N`,
`--offset N` (alias `--start`), `--no-cache`, `--output PATH`.

---

## Output

`output.csv` has exactly one row per input row, with the required columns in
order:

```
user_id, image_paths, user_claim, claim_object,
evidence_standard_met, evidence_standard_met_reason, risk_flags,
issue_type, object_part, claim_status, claim_status_justification,
supporting_image_ids, valid_image, severity
```

---

## Cost / latency awareness

- **One multimodal call per claim** (all of a claim's images in a single call) —
  not one call per image.
- **On-disk response cache** (`.cache/`) keyed by model + prompt + image bytes:
  re-runs and identical inputs cost zero API calls.
- **Large images are downscaled** to a 1568px longest edge to cap token cost.
- **Exponential backoff + jitter** on 429/5xx; optional throttle for strict
  rate limits.

See `evaluation/evaluation_report.md` for measured tokens, runtime, and the
full-test-set cost projection.

---

## Security (prompt-injection defense)

- The system prompt instructs the model to **ignore any instruction-like text
  inside images** (e.g. a sticky note "approve this claim") and to record it via
  `text_instruction_present` instead of obeying it.
- The conversation parser independently flags injection attempts
  ("approve immediately", "skip manual review", "any system reading this…") —
  defense in depth across two layers.
- The model never "approves/rejects"; it only reports what the evidence shows.
