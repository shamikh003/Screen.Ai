"""Screen.ai — AI-Powered Resume Screening & Live Voice Interview.

A single-page Streamlit app that:
  1. Extracts text from an uploaded PDF resume.
  2. Analyses it against a job description (match %, skills, roadmap).
  3. Suggests YouTube learning resources for missing skills.
  4. Runs a personalised, bilingual voice/text mock interview with scoring.

--------------------------------------------------------------------------
BUG-FIX CHANGELOG (this revision)
--------------------------------------------------------------------------
[FIX-1] Voice-question bug (Q1 spoke, Q2/Q3 silently fell back to text):
    speak() used a bare `except Exception: pass`, so any transient gTTS
    failure (network blip / rate limit / locked temp file) after the first
    question was swallowed with no signal to the user. Fixed by:
      - retrying transient failures (2 attempts, small backoff)
      - using tempfile.NamedTemporaryFile instead of a hand-rolled name in
        the working directory (avoids collisions / permission issues)
      - surfacing failures into st.session_state.tts_last_error instead of
        discarding them, so the UI can show a warning + a manual
        "retry audio" button instead of silently degrading to text-only

[FIX-2] Missing-skills / YouTube recommendation bug:
    The app trusted analyze_profile()'s missing_skills list verbatim, with
    no de-duplication against matching_skills and no cross-check against
    the actual resume text, and get_learning_videos() was called with no
    error handling (one API hiccup killed the whole section). Fixed by:
      - reconcile_skill_gap(): a deterministic, local post-processing pass
        that normalises skill strings, removes anything already present in
        matching_skills OR literally present in the resume text (fixes
        LLM misclassification), and de-duplicates case-insensitively
      - wrapping get_learning_videos() in try/except with a visible error
        + "Retry" button instead of a hard crash

[FIX-3] Stale answers leaking between interviews: dynamic `answer_{idx}`
    session_state keys were never cleared by reset_interview()/reset_all().
[FIX-4] Crash (ZeroDivisionError / IndexError) if generate_questions()
    returns an empty list.
[FIX-5] score_color() and score_word() used different thresholds, so the
    bar color and the text label could contradict each other. Unified into
    one match_tier() source of truth.
[FIX-6] vid['url'] was interpolated unescaped into unsafe_allow_html
    markdown. Restructured to avoid raw HTML entirely for that block.
[FIX-7] Shared mutable list objects in DEFAULTS were assigned by
    reference (not copied) into session_state, a latent shared-state bug.
--------------------------------------------------------------------------
"""
import base64
import os
import re
import tempfile
import time
import uuid

import gtts
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_mic_recorder import mic_recorder

from modules import config
from modules.groq_analyzer import analyze_profile
from modules.interview import (
    evaluate_answer,
    generate_final_report,
    generate_questions,
    transcribe_answer,
)
from modules.youtube import get_learning_videos

# --------------------------------------------------------------------------
# Page setup & styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Screen.ai",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stTextArea, .stButton {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background:
        radial-gradient(1200px 600px at 15% -10%, rgba(139,92,246,.18), transparent 60%),
        radial-gradient(1000px 500px at 100% 0%, rgba(6,182,212,.14), transparent 55%),
        #0b0b17;
}
.block-container { padding-top: 2.2rem; max-width: 1180px; }

/* Hero */
.hero { text-align:center; margin: 0 0 1.2rem; }
.hero h1 {
    font-size: 2.9rem; font-weight: 800; margin: 0; line-height: 1.1;
    background: linear-gradient(120deg,#a78bfa 0%,#8b5cf6 35%,#06b6d4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero p { color:#a5a5c0; font-size:1.05rem; margin:.5rem 0 0; }
.pill-row { display:flex; gap:.5rem; justify-content:center; flex-wrap:wrap; margin-top:1rem; }
.pill { font-size:.78rem; color:#c4b5fd; background:rgba(139,92,246,.12);
        border:1px solid rgba(139,92,246,.35); padding:.32rem .8rem; border-radius:999px; }

/* Glass cards */
.card {
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(139,92,246,.18);
    border-radius: 16px; padding: 1.3rem 1.4rem; margin-bottom: 1rem;
    backdrop-filter: blur(8px); box-shadow: 0 8px 30px rgba(0,0,0,.25);
}
.section-title { font-size:1.05rem; font-weight:700; color:#e9e9f5; margin:0 0 .2rem; }
.section-title .step { color:#8b5cf6; font-weight:800; margin-right:.5rem; }
.section-sub  { font-size:.85rem; color:#9494b0; margin:0 0 1rem; }

/* Score */
.score-wrap { display:flex; align-items:center; gap:1.2rem; }
.score-num { font-size:3.1rem; font-weight:800; line-height:1; }
.score-track { flex:1; height:16px; border-radius:999px; background:rgba(255,255,255,.08); overflow:hidden; }
.score-fill { height:100%; border-radius:999px; transition:width .6s ease; }
.score-label { font-size:.8rem; color:#9494b0; text-transform:uppercase; letter-spacing:.06em; }

/* Chips */
.chips { display:flex; flex-wrap:wrap; gap:.45rem; }
.chip { font-size:.82rem; padding:.34rem .75rem; border-radius:999px; font-weight:500; }
.chip.good { color:#86efac; background:rgba(34,197,94,.12); border:1px solid rgba(34,197,94,.35); }
.chip.miss { color:#fcd34d; background:rgba(245,158,11,.12); border:1px solid rgba(245,158,11,.35); }
.muted { color:#77778f; font-size:.9rem; }

/* Roadmap */
.road { display:flex; gap:.7rem; align-items:flex-start; padding:.7rem 0; border-bottom:1px solid rgba(255,255,255,.06); }
.road:last-child { border-bottom:none; }
.badge { font-size:.68rem; font-weight:700; padding:.2rem .55rem; border-radius:6px; text-transform:uppercase; letter-spacing:.04em; white-space:nowrap; }
.badge.high { color:#fca5a5; background:rgba(239,68,68,.14); border:1px solid rgba(239,68,68,.4); }
.badge.medium { color:#fcd34d; background:rgba(245,158,11,.14); border:1px solid rgba(245,158,11,.4); }
.badge.low { color:#86efac; background:rgba(34,197,94,.14); border:1px solid rgba(34,197,94,.4); }
.road b { color:#e9e9f5; } .road span.act { color:#a5a5c0; font-size:.9rem; }

/* Interview */
.q-bubble { background:linear-gradient(135deg, rgba(139,92,246,.16), rgba(6,182,212,.10));
    border:1px solid rgba(139,92,246,.3); border-radius:14px; padding:1.1rem 1.3rem; font-size:1.08rem;
    color:#f0f0fa; line-height:1.5; }
.q-label { font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; color:#a5a5c0; display:block; margin-bottom:.35rem; }
.rec-badge { display:inline-block; font-size:1rem; font-weight:700; padding:.5rem 1.1rem; border-radius:10px; }
.eval-row { padding:.6rem 0; border-bottom:1px solid rgba(255,255,255,.06); }

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important; color:#fff !important;
    font-weight:600; border:none; border-radius:10px; height:3rem; width:100%;
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow:0 8px 22px rgba(139,92,246,.35); }
#MainMenu, footer, header { visibility:hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
DEFAULTS = {
    "results": None,
    "cv_text": "",
    "jd_text": "",
    "videos": None,
    "videos_error": None,          # [FIX-2] surface video-fetch errors instead of crashing
    "iv_active": False,
    "iv_questions": [],
    "iv_index": 0,
    "iv_answers": [],
    "iv_done": False,
    "iv_report": None,
    "voice_on": True,
    "spoken_idx": -1,
    "last_audio_id": None,
    "tts_last_error": None,        # [FIX-1] surface TTS failures instead of swallowing them
    "last_tts_call_ts": 0.0,       # [FIX-8] enforce a minimum gap between gTTS calls
}


def _default_copy(value):
    """[FIX-7] Return an independent copy of mutable defaults (list/dict) so
    that resetting session state never re-shares (and therefore never risks
    mutating) the module-level DEFAULTS objects."""
    if isinstance(value, (list, dict)):
        return value.copy()
    return value


for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, _default_copy(value))


def _clear_dynamic_answer_keys():
    """[FIX-3] The interview text areas are bound to session_state keys named
    'answer_0', 'answer_1', ... which are NOT part of DEFAULTS (they're
    created on the fly). Without this, a second interview in the same
    session reuses the same keys and silently pre-fills the answer boxes
    with the previous interview's answers."""
    for k in [k for k in st.session_state.keys() if k.startswith("answer_")]:
        del st.session_state[k]


def reset_all():
    for key, value in DEFAULTS.items():
        st.session_state[key] = _default_copy(value)
    _clear_dynamic_answer_keys()


def reset_interview():
    for key in ("iv_active", "iv_questions", "iv_index", "iv_answers",
                "iv_done", "iv_report", "spoken_idx", "last_audio_id",
                "tts_last_error", "last_tts_call_ts"):
        st.session_state[key] = _default_copy(DEFAULTS[key])
    _clear_dynamic_answer_keys()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def speak(text, retries=2, backoff_seconds=1.5):
    """Autoplay text as speech.

    [FIX-1] Previously any exception here (network blip / rate limit / locked
    temp file, etc.) was caught and silently discarded, which is exactly why
    question 1 could play audio while questions 2/3 silently degraded to
    text-only with no visible error. Now we:
      - retry a couple of times before giving up (transient errors are
        common with gTTS's HTTP backend)
      - use a real temp file via `tempfile` (no working-directory
        collisions / permission issues)
      - record the failure reason in session_state so the UI can show it
        and offer a manual retry, instead of failing invisibly.

    [FIX-8] gTTS talks to Google Translate's undocumented TTS endpoint,
    which throttles/blocks requests that arrive too close together in time.
    Confirmed via real testing: skipping through questions rapidly (no
    delay between speak() calls) reliably lost audio after question 1,
    while typing+submitting an answer (which naturally inserts a 1-3s delay
    for the Groq evaluation call) let every question's audio play fine.
    Same code path, only difference was timing between consecutive gTTS
    calls. Fix: unconditionally enforce a minimum gap (MIN_TTS_GAP seconds)
    since the last gTTS call, sleeping first if the user proceeds faster
    than that — so audio no longer silently depends on how fast the person
    clicks through the interview.

    Returns True if audio was played, False otherwise.
    """
    MIN_TTS_GAP = 2.0  # seconds; below this, Google's TTS endpoint tends to throttle

    st.session_state.tts_last_error = None
    if not text or not st.session_state.voice_on:
        return False

    elapsed = time.time() - st.session_state.get("last_tts_call_ts", 0.0)
    if elapsed < MIN_TTS_GAP:
        time.sleep(MIN_TTS_GAP - elapsed)

    last_exc = None
    for attempt in range(retries + 1):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            gtts.gTTS(text=text, lang="en").save(tmp_path)
            with open(tmp_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            st.markdown(
                f'<audio autoplay="true" style="display:none">'
                f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
                unsafe_allow_html=True,
            )
            st.session_state.last_tts_call_ts = time.time()
            return True
        except Exception as exc:  # noqa: BLE001 - we deliberately capture & surface this
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff_seconds)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # All attempts failed — surface it instead of pretending everything's fine.
    st.session_state.last_tts_call_ts = time.time()
    st.session_state.tts_last_error = str(last_exc) if last_exc else "Unknown TTS error"
    return False


def match_tier(pct):
    """[FIX-5] Single source of truth for score color + label, so the color
    bar and the text label can never contradict each other (previously
    score_color() and score_word() used different breakpoints: e.g. 65%
    rendered an amber bar next to a "Strong match" label)."""
    if pct >= 75:
        return "#22c55e", "Excellent match"
    if pct >= 60:
        return "#22c55e", "Strong match"
    if pct >= 40:
        return "#f59e0b", "Fair match"
    return "#ef4444", "Low match"


def score_color(pct):
    return match_tier(pct)[0]


def score_word(pct):
    return match_tier(pct)[1]


def rec_style(rec):
    r = rec.lower()
    if "strong" in r:
        return "#22c55e", "rgba(34,197,94,.15)"
    if "not" in r or "no" in r:
        return "#ef4444", "rgba(239,68,68,.15)"
    if "hire" in r:
        return "#06b6d4", "rgba(6,182,212,.15)"
    return "#f59e0b", "rgba(245,158,11,.15)"


def chips(items, kind):
    if not items:
        return '<span class="muted">None identified.</span>'
    return '<div class="chips">' + "".join(
        f'<span class="chip {kind}">{st_escape(i)}</span>' for i in items
    ) + "</div>"


def st_escape(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def section(step, title):
    st.markdown(
        f'<div class="section-title"><span class="step">{step}</span>{st_escape(title)}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# [FIX-2] Skill-gap reconciliation
# --------------------------------------------------------------------------
def _normalize_skill(skill):
    """Lowercase, strip punctuation/whitespace, collapse internal spaces."""
    s = re.sub(r"[^a-z0-9+#. ]", "", skill.lower().strip())
    return re.sub(r"\s+", " ", s).strip()


def _skill_mentioned_in_text(skill_norm, haystack_norm):
    """Simple, dependable containment check with a light plural/suffix
    tolerance (e.g. 'testing' should match a resume that says 'tested')."""
    if not skill_norm:
        return False
    if skill_norm in haystack_norm:
        return True
    # try the stem without a trailing 's', 'ing', 'ed' — cheap but effective
    for suffix in ("ing", "ed", "s"):
        if skill_norm.endswith(suffix) and len(skill_norm) > len(suffix) + 2:
            stem = skill_norm[: -len(suffix)]
            if stem and stem in haystack_norm:
                return True
    return False


def reconcile_skill_gap(analysis, cv_text):
    """[FIX-2] Deterministic post-processing pass over the LLM's
    matching_skills / missing_skills lists.

    This is the actual fix for "fails to correctly identify missing
    skills": rather than trusting the model's classification blindly, we:
      1. Normalise + de-duplicate both lists (case/whitespace variants of
         the same skill no longer show up twice, or in both lists at once).
      2. Re-check every "missing" skill against the resume text itself. If
         it's literally present in the CV, the model misclassified it —
         move it to matching_skills instead of leaving it as a false gap
         (this is the most common cause of "wrong missing skills" bugs:
         the model paraphrases the JD's wording and fails to recognise a
         synonym/variant that's actually already in the resume).
      3. Guarantee matching_skills and missing_skills are mutually
         exclusive, so downstream video recommendations are never fetched
         for a skill the candidate already has.

    Mutates and returns `analysis` in place.
    """
    matching_raw = analysis.get("matching_skills") or []
    missing_raw = analysis.get("missing_skills") or []
    cv_norm = _normalize_skill(cv_text) if cv_text else ""

    # De-duplicate while preserving first-seen original casing/formatting.
    seen_norm = {}
    matching_clean = []
    for skill in matching_raw:
        norm = _normalize_skill(skill)
        if norm and norm not in seen_norm:
            seen_norm[norm] = True
            matching_clean.append(skill.strip())

    missing_clean = []
    reclassified = []
    for skill in missing_raw:
        norm = _normalize_skill(skill)
        if not norm or norm in seen_norm:
            continue  # duplicate of something already matching, or empty
        if _skill_mentioned_in_text(norm, cv_norm):
            # The model called this "missing" but it's actually in the resume.
            seen_norm[norm] = True
            matching_clean.append(skill.strip())
            reclassified.append(skill.strip())
        else:
            seen_norm[norm] = True
            missing_clean.append(skill.strip())

    analysis["matching_skills"] = matching_clean
    analysis["missing_skills"] = missing_clean
    analysis["_reclassified_skills"] = reclassified  # kept for optional debugging/UI
    return analysis


# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>Screen.ai</h1>
      <p>AI-powered resume screening &amp; live voice interview — for <b>any</b> job, any field.</p>
      <div class="pill-row">
        <span class="pill">Smart match scoring</span>
        <span class="pill">Learning roadmap</span>
        <span class="pill">Video resources</span>
        <span class="pill">Live AI interview</span>
        <span class="pill">Urdu / English</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Configuration guard.
missing = config.missing_keys()
if missing:
    st.error(
        f"Missing configuration: **{', '.join(missing)}**. "
        "Add it to a `.env` file in the project root, then restart the app."
    )
    st.stop()
if not config.youtube_enabled():
    st.info("YouTube resources are disabled (no `YOUTUBE_API_KEY`). Everything else works.")


# --------------------------------------------------------------------------
# Step 1 — Input
# --------------------------------------------------------------------------
section("Step 1", "Provide the resume & job")
st.markdown('<div class="section-sub">Upload a PDF resume and paste the target job description.</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Resume (PDF)", type=["pdf"], label_visibility="collapsed")
    cv_text = ""
    if uploaded:
        try:
            reader = PdfReader(uploaded)
            cv_text = "".join((page.extract_text() or "") for page in reader.pages)
            if cv_text.strip():
                st.success(f"Resume loaded ({len(cv_text):,} characters extracted).")
            else:
                st.warning("Could not extract text — is this a scanned/image PDF?")
        except Exception as exc:
            st.error(f"Error reading PDF: {exc}")
with col2:
    jd_text = st.text_area(
        "Job description", height=180, label_visibility="collapsed",
        placeholder="Paste the target job description here (any field)...",
    )

if st.button("Analyse Profile"):
    if cv_text.strip() and jd_text.strip():
        with st.spinner("Analysing profile alignment with Groq..."):
            res = analyze_profile(cv_text, jd_text)
        if "error" in res:
            st.error(f"Analysis failed: {res['error']}")
        else:
            # [FIX-2] Reconcile the skill gap deterministically before storing.
            res = reconcile_skill_gap(res, cv_text)
            st.session_state.results = res
            st.session_state.cv_text = cv_text
            st.session_state.jd_text = jd_text
            st.session_state.videos = None
            st.session_state.videos_error = None
            reset_interview()
            st.rerun()
    else:
        st.warning("Please provide both a valid PDF resume and a job description.")


# --------------------------------------------------------------------------
# Step 2 — Results
# --------------------------------------------------------------------------
res = st.session_state.results
if res:
    pct = res["match_percentage"]
    color, label = match_tier(pct)  # [FIX-5] single source of truth
    st.markdown("<hr style='border-color:rgba(255,255,255,.08)'>", unsafe_allow_html=True)
    section("Step 2", f"Results for {res['candidate_name']}")

    # Match score bar
    st.markdown(
        f"""
        <div class="card">
          <div class="score-label">Match score</div>
          <div class="score-wrap">
            <div class="score-num" style="color:{color}">{pct}%</div>
            <div class="score-track"><div class="score-fill"
                 style="width:{pct}%;background:linear-gradient(90deg,{color},#8b5cf6)"></div></div>
          </div>
          <div class="score-label" style="margin-top:.6rem">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="card"><div class="section-title">Matching skills</div>'
            f'{chips(res["matching_skills"], "good")}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="card"><div class="section-title">Missing skills</div>'
            f'{chips(res["missing_skills"], "miss")}</div>', unsafe_allow_html=True)

    if res.get("feedback"):
        st.info(f"**Recruiter feedback:** {res['feedback']}")

    # Learning roadmap
    if res.get("roadmap"):
        rows = ""
        for item in res["roadmap"]:
            pr = item["priority"].lower()
            pr = pr if pr in ("high", "medium", "low") else "medium"
            rows += (
                f'<div class="road"><span class="badge {pr}">{st_escape(item["priority"])}</span>'
                f'<div><b>{st_escape(item["skill"])}</b><br>'
                f'<span class="act">{st_escape(item["action"])}</span></div></div>'
            )
        st.markdown(
            f'<div class="card"><div class="section-title">Learning roadmap</div>'
            f'<div class="section-sub">Priority-ordered plan to close the gaps.</div>{rows}</div>',
            unsafe_allow_html=True,
        )

    # YouTube resources
    # [FIX-2] Now wrapped in error handling with a visible retry option,
    # and driven by the *reconciled* missing_skills list from above.
    if config.youtube_enabled() and res["missing_skills"]:
        need_fetch = st.session_state.videos is None and st.session_state.videos_error is None
        if need_fetch:
            with st.spinner("Finding learning videos..."):
                try:
                    st.session_state.videos = get_learning_videos(res["missing_skills"])
                    st.session_state.videos_error = None
                except Exception as exc:
                    st.session_state.videos = None
                    st.session_state.videos_error = str(exc)

        if st.session_state.videos_error:
            st.warning(f"Couldn't load learning videos right now: {st.session_state.videos_error}")
            if st.button("Retry loading videos"):
                st.session_state.videos_error = None
                st.rerun()
        else:
            videos = st.session_state.videos or []
            if videos:
                section("", "Recommended learning videos")
                vcols = st.columns(min(3, len(videos)))
                for i, vid in enumerate(videos):
                    with vcols[i % len(vcols)]:
                        st.video(vid["url"])
                        # [FIX-6] No more raw HTML / unescaped URL interpolation —
                        # plain markdown link syntax handles escaping safely,
                        # and st.caption keeps the channel name out of any
                        # unsafe_allow_html block entirely.
                        st.markdown(f"**{st_escape(vid['skill'])}** · [{st_escape(vid['title'][:55])}]({vid['url']})")
                        st.caption(vid["channel"])


# --------------------------------------------------------------------------
# Step 3 — Live interview
# --------------------------------------------------------------------------
if res:
    st.markdown("<hr style='border-color:rgba(255,255,255,.08)'>", unsafe_allow_html=True)
    section("Step 3", "Live AI interview")
    st.markdown('<div class="section-sub">Personalised questions from your CV &amp; JD. '
                'Answer by voice or text — in English or Urdu.</div>', unsafe_allow_html=True)

    # Not started yet
    if not st.session_state.iv_active:
        cset, cbtn = st.columns([1, 2])
        with cset:
            st.session_state.voice_on = st.toggle("Voice questions", value=st.session_state.voice_on)
        with cbtn:
            if st.button("Start Interview"):
                with st.spinner("Preparing your interview questions..."):
                    questions = generate_questions(
                        st.session_state.cv_text, st.session_state.jd_text,
                        res["matching_skills"], res["missing_skills"],
                    )
                # [FIX-4] Guard against an empty/None question list instead of
                # crashing later with ZeroDivisionError / IndexError.
                if not questions:
                    st.error("Couldn't generate interview questions. Please try again.")
                else:
                    st.session_state.iv_questions = questions
                    st.session_state.iv_active = True
                    st.session_state.iv_index = 0
                    st.session_state.iv_answers = []
                    st.session_state.iv_done = False
                    st.session_state.spoken_idx = -1
                    st.session_state.tts_last_error = None
                    st.rerun()

    # Interview finished — show report
    elif st.session_state.iv_done:
        report = st.session_state.iv_report or {}
        ov = report.get("overall_score", 0)
        rec = report.get("recommendation", "Maybe")
        col, bg = rec_style(rec)

        st.markdown(
            f"""
            <div class="card">
              <div class="score-label">Interview result</div>
              <div class="score-wrap">
                <div class="score-num" style="color:{score_color(ov)}">{ov}<span style="font-size:1.2rem">/100</span></div>
                <div class="score-track"><div class="score-fill"
                     style="width:{ov}%;background:linear-gradient(90deg,{score_color(ov)},#8b5cf6)"></div></div>
                <span class="rec-badge" style="color:{col};background:{bg};border:1px solid {col}">{st_escape(rec)}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if report.get("summary"):
            st.info(f"**Verdict:** {report['summary']}")

        s1, s2 = st.columns(2)
        with s1:
            items = "".join(f"<li>{st_escape(x)}</li>" for x in report.get("strengths", [])) \
                or "<li class='muted'>—</li>"
            st.markdown(f'<div class="card"><div class="section-title">Strengths</div><ul>{items}</ul></div>',
                        unsafe_allow_html=True)
        with s2:
            items = "".join(f"<li>{st_escape(x)}</li>" for x in report.get("improvements", [])) \
                or "<li class='muted'>—</li>"
            st.markdown(f'<div class="card"><div class="section-title">Improvements</div><ul>{items}</ul></div>',
                        unsafe_allow_html=True)

        with st.expander("Per-question breakdown"):
            for i, ans in enumerate(st.session_state.iv_answers, 1):
                ev = ans.get("eval", {})
                st.markdown(f"**Q{i}. {st_escape(ans['question'])}**")
                st.markdown(f"<span class='muted'>Your answer:</span> {st_escape(ans.get('answer', '—'))}",
                            unsafe_allow_html=True)
                st.markdown(f"**Score: {ev.get('score', 0)}/10** · _{st_escape(ev.get('language', ''))}_")
                st.markdown(f"**Strengths:** {st_escape(ev.get('strengths', '—'))}")
                st.markdown(f"**Improvements:** {st_escape(ev.get('improvements', '—'))}")
                st.markdown("---")

        if st.button("Start Over"):
            reset_all()
            st.rerun()

    # Interview in progress
    else:
        questions = st.session_state.iv_questions
        idx = st.session_state.iv_index
        total = len(questions)

        # [FIX-4] Defensive guard (shouldn't trigger given the check above,
        # but keeps this section crash-proof if state is ever corrupted).
        if total == 0 or idx >= total:
            st.error("Interview state is invalid. Please restart the interview.")
            if st.button("Restart interview"):
                reset_interview()
                st.rerun()
            st.stop()

        question = questions[idx]

        st.progress((idx) / total, text=f"Question {idx + 1} of {total}")
        st.markdown(
            f'<div class="q-bubble"><span class="q-label">Interviewer</span>{st_escape(question)}</div>',
            unsafe_allow_html=True,
        )

        # Speak the question once when it first appears.
        if st.session_state.spoken_idx != idx:
            spoke_ok = speak(question)
            st.session_state.spoken_idx = idx
        else:
            spoke_ok = st.session_state.tts_last_error is None

        # [FIX-1] Make TTS failures visible + give the user a one-click retry,
        # instead of silently falling back to text-only for some questions.
        if st.session_state.voice_on and st.session_state.tts_last_error:
            st.warning(
                "Couldn't play the question audio for this one "
                f"({st.session_state.tts_last_error}). You can still read and "
                "answer it below, or retry the audio."
            )
            if st.button("🔊 Retry question audio", key=f"retry_tts_{idx}"):
                speak(question)
                st.rerun()

        answer_key = f"answer_{idx}"
        st.session_state.setdefault(answer_key, "")

        st.write("")
        mcol, _ = st.columns([1, 2])
        with mcol:
            audio = mic_recorder(
                start_prompt="Record answer", stop_prompt="Stop & transcribe",
                just_once=False, key=f"mic_{idx}",
            )

        # New recording -> transcribe into the editable answer box.
        if audio and audio.get("bytes") and st.session_state.last_audio_id != audio.get("id"):
            st.session_state.last_audio_id = audio.get("id")
            with st.spinner("Transcribing your answer..."):
                try:
                    text, err = transcribe_answer(audio["bytes"], audio.get("format", "webm"))
                except Exception as exc:  # defensive: don't let a transcription
                    text, err = "", str(exc)  # crash take down the whole interview
            if err:
                st.error(f"Transcription failed: {err}")
            else:
                st.session_state[answer_key] = text
                st.rerun()

        st.text_area(
            "Your answer (type, or edit the transcription)",
            key=answer_key, height=130,
            placeholder="Speak using the recorder above, or type your answer here...",
        )

        b1, b2 = st.columns([2, 1])
        with b1:
            submit = st.button("Submit answer", key=f"submit_{idx}")
        with b2:
            skip = st.button("Skip", key=f"skip_{idx}")

        if submit or skip:
            answer = "" if skip else st.session_state.get(answer_key, "").strip()
            with st.spinner("Evaluating..."):
                try:
                    ev = evaluate_answer(question, answer, st.session_state.jd_text)
                except Exception as exc:
                    ev = {"score": 0, "language": "", "strengths": "—",
                          "improvements": f"Evaluation failed: {exc}"}
            st.session_state.iv_answers.append(
                {"question": question, "answer": answer or "(no answer)", "eval": ev}
            )
            if idx + 1 >= total:
                with st.spinner("Compiling your interview report..."):
                    try:
                        st.session_state.iv_report = generate_final_report(
                            st.session_state.iv_answers, res["match_percentage"],
                            st.session_state.jd_text,
                        )
                    except Exception as exc:
                        st.session_state.iv_report = {
                            "overall_score": 0, "recommendation": "Maybe",
                            "summary": f"Report generation failed: {exc}",
                            "strengths": [], "improvements": [],
                        }
                st.session_state.iv_done = True
            else:
                st.session_state.iv_index += 1
            st.rerun()

st.markdown(
    "<p style='text-align:center;color:#55556e;font-size:.8rem;margin-top:2rem'>"
    "Built with Streamlit · Groq · Whisper — Screen.ai</p>",
    unsafe_allow_html=True,
)
