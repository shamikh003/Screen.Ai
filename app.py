"""Screen.ai — AI-Powered Resume Screening & Live Voice Interview.

A single-page Streamlit app that:
  1. Extracts text from an uploaded PDF resume.
  2. Analyses it against a job description (match %, skills, roadmap).
  3. Suggests YouTube learning resources for missing skills.
  4. Runs a personalised, bilingual voice/text mock interview with scoring.
"""
import base64
import os
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
    "iv_active": False,
    "iv_questions": [],
    "iv_index": 0,
    "iv_answers": [],
    "iv_done": False,
    "iv_report": None,
    "voice_on": True,
    "spoken_idx": -1,
    "last_audio_id": None,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_all():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def reset_interview():
    for key in ("iv_active", "iv_questions", "iv_index", "iv_answers",
                "iv_done", "iv_report", "spoken_idx", "last_audio_id"):
        st.session_state[key] = DEFAULTS[key]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def speak(text):
    """Autoplay text as speech (best-effort; silent on failure)."""
    if not text or not st.session_state.voice_on:
        return
    path = f"_tts_{uuid.uuid4().hex}.mp3"
    try:
        gtts.gTTS(text=text, lang="en").save(path)
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        st.markdown(
            f'<audio autoplay="true" style="display:none">'
            f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass
    finally:
        if os.path.exists(path):
            os.remove(path)


def score_color(pct):
    if pct >= 70:
        return "#22c55e"
    if pct >= 40:
        return "#f59e0b"
    return "#ef4444"


def score_word(pct):
    if pct >= 75:
        return "Excellent match"
    if pct >= 60:
        return "Strong match"
    if pct >= 40:
        return "Fair match"
    return "Low match"


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
            st.session_state.results = res
            st.session_state.cv_text = cv_text
            st.session_state.jd_text = jd_text
            st.session_state.videos = None
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
    st.markdown("<hr style='border-color:rgba(255,255,255,.08)'>", unsafe_allow_html=True)
    section("Step 2", f"Results for {res['candidate_name']}")

    # Match score bar
    st.markdown(
        f"""
        <div class="card">
          <div class="score-label">Match score</div>
          <div class="score-wrap">
            <div class="score-num" style="color:{score_color(pct)}">{pct}%</div>
            <div class="score-track"><div class="score-fill"
                 style="width:{pct}%;background:linear-gradient(90deg,{score_color(pct)},#8b5cf6)"></div></div>
          </div>
          <div class="score-label" style="margin-top:.6rem">{score_word(pct)}</div>
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
    if config.youtube_enabled() and res["missing_skills"]:
        if st.session_state.videos is None:
            with st.spinner("Finding learning videos..."):
                st.session_state.videos = get_learning_videos(res["missing_skills"])
        videos = st.session_state.videos or []
        if videos:
            section("", "Recommended learning videos")
            vcols = st.columns(min(3, len(videos)))
            for i, vid in enumerate(videos):
                with vcols[i % len(vcols)]:
                    st.video(vid["url"])
                    st.markdown(
                        f"**{st_escape(vid['skill'])}** · "
                        f"[{st_escape(vid['title'][:55])}]({vid['url']})"
                        f"<br><span class='muted'>{st_escape(vid['channel'])}</span>",
                        unsafe_allow_html=True,
                    )


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
                    st.session_state.iv_questions = generate_questions(
                        st.session_state.cv_text, st.session_state.jd_text,
                        res["matching_skills"], res["missing_skills"],
                    )
                st.session_state.iv_active = True
                st.session_state.iv_index = 0
                st.session_state.iv_answers = []
                st.session_state.iv_done = False
                st.session_state.spoken_idx = -1
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
        question = questions[idx]

        st.progress((idx) / total, text=f"Question {idx + 1} of {total}")
        st.markdown(
            f'<div class="q-bubble"><span class="q-label">Interviewer</span>{st_escape(question)}</div>',
            unsafe_allow_html=True,
        )

        # Speak the question once when it first appears.
        if st.session_state.spoken_idx != idx:
            speak(question)
            st.session_state.spoken_idx = idx

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
                text, err = transcribe_answer(audio["bytes"], audio.get("format", "webm"))
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
                ev = evaluate_answer(question, answer, st.session_state.jd_text)
            st.session_state.iv_answers.append(
                {"question": question, "answer": answer or "(no answer)", "eval": ev}
            )
            if idx + 1 >= total:
                with st.spinner("Compiling your interview report..."):
                    st.session_state.iv_report = generate_final_report(
                        st.session_state.iv_answers, res["match_percentage"],
                        st.session_state.jd_text,
                    )
                st.session_state.iv_done = True
            else:
                st.session_state.iv_index += 1
            st.rerun()

st.markdown(
    "<p style='text-align:center;color:#55556e;font-size:.8rem;margin-top:2rem'>"
    "Built with Streamlit · Groq · Whisper — Screen.ai</p>",
    unsafe_allow_html=True,
)
