# Multi-Modal Evidence Review

Verifies damage claims for **cars, laptops, and packages** by reasoning over the
submitted images, the claim conversation, user history, and minimum evidence
requirements. For each claim it decides whether the images **support**,
**contradict**, or provide **not_enough_information**, and emits structured
fields (issue type, object part, severity, risk flags, supporting image ids,
evidence sufficiency).

The **images are the source of truth**; the conversation defines *what to check*;
user history only adds *risk context* and never overrides clear visual evidence.
Hardened against prompt injection in both the chat **and** inside the images.

---

## Setup

Requires **Python 3.10+** and a **Gemini API key** (free at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey)).

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then add your key to .env (gitignored)
# .env:  PROVIDER=gemini   GEMINI_API_KEY=your_key_here
```

Secrets are read from environment variables only. The provider is pluggable via
`PROVIDER` (`gemini` is the implemented primary).

## Run

```bash
python main.py --mode test --limit 3     # cheap smoke test
python main.py --mode test               # -> dataset/output.csv (all claims)
python main.py --mode sample             # -> code/evaluation/sample_output.csv
python evaluation/main.py                # metrics + strategy comparison + report
```

Flags: `--strategy {context,two_stage,evidence_first}`, `--limit N`, `--no-cache`.

## How it works

```
claim_parser → user_risk → evidence_checker → image_analyzer (VLM, cached) → decision_engine
```

Only the **visual judgment** uses the model; claim parsing, history risk,
requirement selection, enum validation, and flag-merging are deterministic and
reproducible. One multimodal call per claim, an on-disk response cache, and
large-image downscaling keep cost and latency low. Module-level detail is in
[`code/README.md`](./code/README.md); measured tokens/cost and the strategy
comparison are in `code/evaluation/evaluation_report.md`.

## Layout

| Path | What |
|---|---|
| `code/` | Solution source (see `code/README.md`) |
| `code/main.py` | Pipeline entry point |
| `code/evaluation/` | Metrics + generated `evaluation_report.md` |
| `dataset/` | Input CSVs and images |
| `dataset/output.csv` | Final predictions |

## Submission

- **`code.zip`** — the `code/` folder (incl. `evaluation/`), README, prompts/configs.
- **`output.csv`** — predictions for every row of `dataset/claims.csv`.
- **`chat_transcript`** — `~/hackerrank_orchestrate/log.txt`.

> ⚠️ Never commit secrets. `.env`, `.venv/`, and `.cache/` are gitignored.
