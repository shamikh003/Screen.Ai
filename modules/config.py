"""Central configuration for Screen.ai.

All secrets are loaded from environment variables / .env only.
Nothing sensitive is ever hard-coded in the source tree.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root regardless of the current working directory,
# so the app works no matter where it is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()  # also honour any .env in the current working directory

# ---- Secrets (env / .env locally, st.secrets on Streamlit Cloud) ---------
def _get_secret(name):
    """Read a secret from the environment (.env / OS) first, then fall back
    to Streamlit Cloud's st.secrets. Never hard-coded in the source tree."""
    val = os.getenv(name)
    if val:
        return val.strip()
    # Streamlit Community Cloud exposes dashboard secrets via st.secrets,
    # not as environment variables — guarded so local runs without a
    # secrets.toml don't raise.
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return ""


GROQ_API_KEY = _get_secret("GROQ_API_KEY")
YOUTUBE_API_KEY = _get_secret("YOUTUBE_API_KEY")

# ---- Groq endpoints & models --------------------------------------------
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "openai/gpt-oss-20b"          # chat / analysis (JSON-capable)
WHISPER_MODEL = "whisper-large-v3-turbo"  # speech-to-text

# ---- Behaviour tuning -----------------------------------------------------
MAX_INPUT_CHARS = 14000       # trim very long CV/JD text before sending
INTERVIEW_QUESTIONS = 4       # number of questions in a live interview
REQUEST_TIMEOUT = 60          # seconds


def smart_truncate(text, limit=MAX_INPUT_CHARS):
    """[BUG FIX] Trim long text to `limit` characters WITHOUT blindly
    cutting off the tail.

    The previous code did `text[:MAX_INPUT_CHARS]` everywhere. Most resumes
    put their Skills section near the END of the document (after Experience
    / Education), so any CV longer than ~14,000 characters had its skills
    list silently chopped off before the model ever saw it — the model then
    (correctly, given what it was shown) reported those skills as "missing".
    This is very likely the actual root cause of "missing skills detected
    incorrectly".

    Fix: keep the first ~65% and the last ~30% of the allowed budget (with a
    small explicit marker in between), so header/summary AND a trailing
    skills/certifications section both have a good chance of surviving
    truncation, regardless of resume layout.
    """
    text = text or ""
    if len(text) <= limit:
        return text
    head_len = int(limit * 0.65)
    marker = "\n...[content trimmed for length]...\n"
    tail_len = max(0, limit - head_len - len(marker))
    return text[:head_len] + marker + (text[-tail_len:] if tail_len else "")


def missing_keys():
    """Return a list of required secrets that are not configured."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    return missing


def youtube_enabled():
    """YouTube resources are optional — only used when a key is present."""
    return bool(YOUTUBE_API_KEY)
