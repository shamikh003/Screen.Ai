"""Thin, dependency-light Groq client.

Uses the OpenAI-compatible REST endpoints via ``requests`` so behaviour is
fully predictable and independent of any SDK version. Handles chat
completions (JSON mode) and Whisper audio transcription.
"""
import json
import re
import time

import requests

from modules import config

_MAX_RETRIES = 2  # extra attempts on transient rate-limit (429) responses


class GroqError(Exception):
    """Raised when the Groq API returns an error or an unusable response."""


def _extract_json(text):
    """Best-effort parse of a JSON object from a model response.

    Reasoning models occasionally wrap JSON in prose or code fences, so we
    fall back to grabbing the outermost ``{...}`` block.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip ```json ... ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Fall back to the first balanced-looking object.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise GroqError("Model did not return valid JSON.")


def _safe_response_json(resp):
    """[BUG FIX] `.json()` raises a raw ValueError/JSONDecodeError (NOT a
    GroqError) if the server ever returns a non-JSON body — e.g. a gateway
    timeout HTML page, or a truncated response during an outage. Previously
    that raw exception was uncaught here, so it would propagate all the way
    up as an unhandled crash instead of the clean {"error": "..."} the rest
    of the app expects. Now it's always converted into a GroqError with
    enough context (status code + a text snippet) to actually debug it."""
    try:
        return resp.json()
    except ValueError as exc:
        snippet = (resp.text or "")[:300]
        raise GroqError(
            f"Groq returned a non-JSON response (HTTP {resp.status_code}): {snippet}"
        ) from exc


def chat_json(messages, temperature=0.3, max_tokens=2500):
    """Call the chat completion endpoint in JSON mode and return a dict."""
    if not config.GROQ_API_KEY:
        raise GroqError("GROQ_API_KEY is not set. Add it to your .env file.")

    url = f"{config.GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    data = _post_with_retry(url, headers, payload)
    if isinstance(data, dict) and data.get("error"):
        raise GroqError(data["error"].get("message", "Unknown Groq error."))

    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GroqError("Unexpected response shape from Groq.") from exc

    # [BUG FIX] If the model's answer got cut off because max_tokens ran
    # out mid-JSON (very possible on a long CV with many missing skills +
    # a full roadmap), the previous code just tried to parse the truncated
    # text and failed with an opaque "Model did not return valid JSON."
    # This detects that specific case, automatically retries ONCE with a
    # bigger budget, and only gives up with a clear message if that also
    # fails — instead of silently producing an incomplete/empty analysis.
    if choice.get("finish_reason") == "length":
        bigger_budget = min(max_tokens * 2, 8000)
        if bigger_budget > max_tokens:
            payload["max_tokens"] = bigger_budget
            data = _post_with_retry(url, headers, payload)
            if isinstance(data, dict) and data.get("error"):
                raise GroqError(data["error"].get("message", "Unknown Groq error."))
            try:
                choice = data["choices"][0]
                content = choice["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise GroqError("Unexpected response shape from Groq.") from exc
            if choice.get("finish_reason") == "length":
                raise GroqError(
                    "The model's response was cut off (too long) even after "
                    "retrying with a larger token budget. Try a shorter CV/JD."
                )

    parsed = _extract_json(content)
    # [BUG FIX] _extract_json() only guarantees valid JSON, not that it's an
    # OBJECT. If the model ever emits a bare JSON list/string/number instead
    # of the requested {...} shape, every caller (analyze_profile,
    # generate_questions, evaluate_answer, generate_final_report) would
    # crash with an uncaught AttributeError on `.get(...)`, since none of
    # them catch anything but GroqError. Centralising the check here means
    # every caller is protected at once instead of relying on each of them
    # to re-check it individually.
    if not isinstance(parsed, dict):
        raise GroqError("Model returned valid JSON but not the expected object shape.")
    return parsed


def _post_with_retry(url, headers, payload):
    """POST JSON, retrying briefly on 429 rate-limit responses."""
    last_error = "Request failed."
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=config.REQUEST_TIMEOUT
            )
        except requests.exceptions.RequestException as exc:
            raise GroqError(f"Network error contacting Groq: {exc}") from exc

        if resp.status_code == 429 and attempt < _MAX_RETRIES:
            wait = _retry_after(resp)
            last_error = "Rate limit reached."
            time.sleep(wait)
            continue
        return _safe_response_json(resp)

    raise GroqError(last_error)


def _retry_after(resp):
    """Seconds to wait before retrying, from the Retry-After header (capped)."""
    try:
        return min(8.0, max(1.0, float(resp.headers.get("retry-after", 2))))
    except (TypeError, ValueError):
        return 2.0


def transcribe(audio_bytes, filename="answer.wav"):
    """Transcribe raw audio bytes to text using Groq Whisper."""
    if not config.GROQ_API_KEY:
        raise GroqError("GROQ_API_KEY is not set. Add it to your .env file.")
    if not audio_bytes:
        raise GroqError("No audio was captured.")

    url = f"{config.GROQ_BASE_URL}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
    ext = filename.rsplit(".", 1)[-1].lower()
    mime = {
        "wav": "audio/wav", "webm": "audio/webm", "mp3": "audio/mpeg",
        "m4a": "audio/mp4", "ogg": "audio/ogg", "flac": "audio/flac",
    }.get(ext, "application/octet-stream")
    files = {"file": (filename, audio_bytes, mime)}
    data = {"model": config.WHISPER_MODEL, "response_format": "json"}

    try:
        resp = requests.post(
            url,
            headers=headers,
            files=files,
            data=data,
            timeout=config.REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException as exc:
        raise GroqError(f"Network error during transcription: {exc}") from exc

    payload = _safe_response_json(resp)
    if isinstance(payload, dict) and payload.get("error"):
        raise GroqError(payload["error"].get("message", "Transcription failed."))

    return (payload.get("text") or "").strip()
