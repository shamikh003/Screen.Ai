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
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GroqError("Unexpected response shape from Groq.") from exc

    return _extract_json(content)


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
        return resp.json()

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

    payload = resp.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise GroqError(payload["error"].get("message", "Transcription failed."))

    return (payload.get("text") or "").strip()
