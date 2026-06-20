"""
image_analyzer.py — The VLM call: send claim context + images, get structured
analysis back.

Design goals:
- Provider-agnostic interface. Gemini (primary) and Amazon Bedrock (Claude
  vision) are implemented; OpenAI/Anthropic are stubbed with clear errors so the
  provider is a config swap.
- Robust structured output via Gemini's response_schema (Pydantic), with a
  defensive JSON fallback if parsing fails.
- Three strategies:
    * "context"        : single claim-aware multimodal call (default, cheapest).
    * "evidence_first" : single call, rubric forces neutral observation first.
    * "two_stage"      : TWO calls — Stage 1 describes images blind (no claim),
                         Stage 2 reconciles the neutral observations with the claim.
- On-disk response cache keyed by (provider, model, strategy, prompts, image
  bytes) so re-runs and duplicate inputs cost nothing.
- Exponential-backoff retries and per-call token/usage accounting.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

import config
import prompts
import utils
from utils import EncodedImage, log


# ---------------------------------------------------------------------------
# Structured-output schemas
# ---------------------------------------------------------------------------

class ImageAssessment(BaseModel):
    image_id: str = ""
    usable: bool = True
    quality_issues: list[str] = Field(default_factory=list)
    shows_claimed_part: bool = False
    visible_content: str = ""
    text_in_image: Optional[str] = None
    damage_visible: str = "unknown"
    supports_decision: bool = False


class ClaimAssessment(BaseModel):
    object_seen: str = "unknown"
    object_matches_claim: bool = True
    images: list[ImageAssessment] = Field(default_factory=list)
    visible_issue_type: str = "unknown"
    object_part: str = "unknown"
    claim_status: str = "not_enough_information"
    evidence_standard_met: bool = False
    valid_image: bool = True
    severity: str = "unknown"
    supporting_image_ids: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    manipulation_suspected: bool = False
    text_instruction_present: bool = False
    evidence_standard_met_reason: str = ""
    claim_status_justification: str = ""


class BlindImage(BaseModel):
    image_id: str = ""
    usable: bool = True
    quality_issues: list[str] = Field(default_factory=list)
    visible_content: str = ""
    object_part_visible: str = "unknown"
    damage_visible: str = "unknown"
    severity_visible: str = "unknown"
    text_in_image: Optional[str] = None


class BlindObservation(BaseModel):
    object_seen: str = "unknown"
    images: list[BlindImage] = Field(default_factory=list)


@dataclass
class _GenResult:
    data: dict
    input_tokens: int = 0
    output_tokens: int = 0
    images_sent: int = 0


@dataclass
class AnalysisResult:
    assessment: dict           # normalized ClaimAssessment as a plain dict
    input_tokens: int = 0
    output_tokens: int = 0
    images_sent: int = 0
    n_calls: int = 1
    cached: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Provider clients
# ---------------------------------------------------------------------------

class GeminiClient:
    """Wraps google-genai for vision + structured JSON output."""

    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Add it to code/.env"
            )
        from google import genai
        from google.genai import types as genai_types
        self._types = genai_types
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.GEMINI_MODEL
        self.thinking_budget = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))

    def _gen_config(self, system_prompt: str, schema):
        types = self._types
        kwargs = dict(
            system_instruction=system_prompt,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=4096,
        )
        if self.thinking_budget >= 0:
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget)
            except Exception:
                pass
        return types.GenerateContentConfig(**kwargs)

    def generate(self, system_prompt: str, user_text: str,
                 images: list[EncodedImage], schema) -> _GenResult:
        types = self._types
        parts = [types.Part.from_text(text=user_text)]
        sent = 0
        for img in images:
            if img.exists and img.data:
                parts.append(types.Part.from_bytes(data=img.data, mime_type=img.mime_type))
                sent += 1

        @utils.retry()
        def _call():
            return self.client.models.generate_content(
                model=self.model, contents=parts,
                config=self._gen_config(system_prompt, schema))

        response = _call()
        in_tok = out_tok = 0
        try:
            um = response.usage_metadata
            in_tok = getattr(um, "prompt_token_count", 0) or 0
            out_tok = (getattr(um, "candidates_token_count", 0) or 0) + \
                      (getattr(um, "thoughts_token_count", 0) or 0)
        except Exception:
            pass
        data = self._parse(response, schema)
        return _GenResult(data=data, input_tokens=in_tok, output_tokens=out_tok,
                          images_sent=sent)

    def _parse(self, response, schema) -> dict:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, BaseModel):
            return parsed.model_dump()
        text = (getattr(response, "text", None) or "").strip()
        return _coerce_json(text, schema)


class BedrockClient:
    """Amazon Bedrock — Claude vision via the Anthropic Messages API on
    bedrock-runtime. Same contract as GeminiClient.generate: text + 0..N images
    in, _GenResult out. The model returns text JSON which _coerce_json parses
    into the existing schema (business logic unchanged).
    """

    def __init__(self):
        import boto3  # lazy: only required when PROVIDER=bedrock
        self.client = boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)
        self.model_id = config.BEDROCK_MODEL_ID

    def generate(self, system_prompt: str, user_text: str,
                 images: list[EncodedImage], schema) -> _GenResult:
        import base64
        # Claude multimodal: a single user turn with text + image content blocks.
        content = [{
            "type": "text",
            "text": user_text + "\n\nReturn ONLY a single valid JSON object "
                                "matching the schema. No prose, no markdown.",
        }]
        sent = 0
        for img in images:
            if img.exists and img.data:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.mime_type or "image/jpeg",
                        "data": base64.b64encode(img.data).decode("ascii"),
                    },
                })
                sent += 1

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
        }

        @utils.retry()
        def _call():
            return self.client.invoke_model(
                modelId=self.model_id, body=json.dumps(body))

        resp = _call()
        payload = json.loads(resp["body"].read())
        # Concatenate any text blocks in the assistant response.
        text = "".join(b.get("text", "") for b in payload.get("content", [])
                       if b.get("type") == "text")
        usage = payload.get("usage", {}) or {}  # default to 0 if absent
        return _GenResult(
            data=_coerce_json(text, schema),
            input_tokens=usage.get("input_tokens", 0) or 0,
            output_tokens=usage.get("output_tokens", 0) or 0,
            images_sent=sent,
        )


class _UnavailableClient:
    def __init__(self, provider: str, reason: str):
        self.provider, self.reason = provider, reason

    def generate(self, *a, **k):
        raise NotImplementedError(
            f"Provider '{self.provider}' is not wired up in this build: {self.reason}. "
            f"Set PROVIDER=gemini (primary) or implement the client."
        )


def get_client():
    if config.PROVIDER == "bedrock":
        return BedrockClient()
    if config.PROVIDER == "openai":
        return _UnavailableClient("openai", "stub only")
    if config.PROVIDER == "anthropic":
        return _UnavailableClient("anthropic", "stub only")
    return GeminiClient()


# ---------------------------------------------------------------------------
# JSON coercion
# ---------------------------------------------------------------------------

def _coerce_json(text: str, schema) -> dict:
    if not text:
        return schema().model_dump()
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        t = t[start:end + 1]
    try:
        return schema(**json.loads(t)).model_dump()
    except Exception as exc:
        log.error("Failed to parse VLM JSON: %s", exc)
        out = schema().model_dump()
        if "claim_status_justification" in out:
            out["claim_status_justification"] = "Model output could not be parsed."
        return out


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_key(strategy: str, system_prompt: str, user_text: str,
               images: list[EncodedImage]) -> str:
    h = hashlib.sha256()
    h.update(config.PROVIDER.encode())
    h.update(config.active_model().encode())
    h.update(strategy.encode())
    h.update(system_prompt.encode("utf-8"))
    h.update(user_text.encode("utf-8"))
    for img in images:
        h.update(img.image_id.encode())
        h.update(hashlib.sha256(img.data).digest() if img.data else b"missing")
    return h.hexdigest()[:32]


def _cache_path(key: str) -> Path:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.CACHE_DIR / f"{key}.json"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_claim(client, *, claim_object: str, parsed_summary: str,
                  customer_statements: str, requirements_text: str,
                  history_context: str, images: list[EncodedImage],
                  adversarial: bool, adversarial_phrases: list[str],
                  strategy: str = "context", use_cache: bool = True) -> AnalysisResult:
    """Build the prompt(s), call the VLM (or cache), return a normalized result."""
    image_ids = [img.image_id for img in images]

    if strategy == "two_stage":
        system_prompt = prompts.build_system_prompt("context")
        # cache key for the whole 2-stage result keyed on the reconcile system +
        # blind user prompt (deterministic per claim).
        cache_seed = "two_stage::" + prompts.build_blind_user_prompt(claim_object, image_ids)
    else:
        system_prompt = prompts.build_system_prompt(strategy)
        cache_seed = prompts.build_user_prompt(
            claim_object=claim_object, parsed_summary=parsed_summary,
            customer_statements=customer_statements, requirements_text=requirements_text,
            history_context=history_context, image_ids=image_ids,
            adversarial=adversarial, adversarial_phrases=adversarial_phrases)

    key = _cache_key(strategy, system_prompt, cache_seed, images)
    cpath = _cache_path(key)
    if use_cache and cpath.exists():
        try:
            payload = json.loads(cpath.read_text(encoding="utf-8"))
            return AnalysisResult(
                assessment=payload["assessment"],
                input_tokens=payload.get("input_tokens", 0),
                output_tokens=payload.get("output_tokens", 0),
                images_sent=payload.get("images_sent", 0),
                n_calls=payload.get("n_calls", 1),
                cached=True,
            )
        except Exception:
            pass

    try:
        if strategy == "two_stage":
            result = _run_two_stage(
                client, claim_object=claim_object, parsed_summary=parsed_summary,
                customer_statements=customer_statements, requirements_text=requirements_text,
                history_context=history_context, images=images, image_ids=image_ids,
                adversarial=adversarial, adversarial_phrases=adversarial_phrases,
                system_prompt=system_prompt)
        else:
            user_text = cache_seed
            gen = client.generate(system_prompt, user_text, images, ClaimAssessment)
            result = AnalysisResult(assessment=gen.data, input_tokens=gen.input_tokens,
                                    output_tokens=gen.output_tokens,
                                    images_sent=gen.images_sent, n_calls=1)
    except Exception as exc:
        log.error("VLM call failed: %s", exc)
        out = ClaimAssessment().model_dump()
        out["claim_status_justification"] = f"Automated analysis failed: {exc}"
        return AnalysisResult(assessment=out, error=str(exc))

    if use_cache and not result.error:
        try:
            cpath.write_text(json.dumps({
                "assessment": result.assessment,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "images_sent": result.images_sent,
                "n_calls": result.n_calls,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return result


def _run_two_stage(client, *, claim_object, parsed_summary, customer_statements,
                   requirements_text, history_context, images, image_ids,
                   adversarial, adversarial_phrases, system_prompt) -> AnalysisResult:
    # Stage 1 — blind perception (no claim/history given).
    blind_sys = prompts.build_blind_system_prompt()
    blind_usr = prompts.build_blind_user_prompt(claim_object, image_ids)
    blind = client.generate(blind_sys, blind_usr, images, BlindObservation)

    # Stage 2 — reconcile neutral observations with the claim.
    blind_json = json.dumps(blind.data, ensure_ascii=False)
    reconcile_usr = prompts.build_reconcile_user_prompt(
        claim_object=claim_object, parsed_summary=parsed_summary,
        customer_statements=customer_statements, requirements_text=requirements_text,
        history_context=history_context, image_ids=image_ids,
        adversarial=adversarial, adversarial_phrases=adversarial_phrases,
        blind_observations_json=blind_json)
    final = client.generate(system_prompt, reconcile_usr, images, ClaimAssessment)

    return AnalysisResult(
        assessment=final.data,
        input_tokens=blind.input_tokens + final.input_tokens,
        output_tokens=blind.output_tokens + final.output_tokens,
        images_sent=blind.images_sent + final.images_sent,
        n_calls=2,
    )
