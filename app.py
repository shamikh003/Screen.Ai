"""
app.py — Screen.ai 
Author: Muhammad Shamikh | Full Stack Developer
"""

import os, io, base64, requests
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import streamlit.components.v1 as components
import PyPDF2
from gtts import gTTS
from modules.groq_analyzer import (
    analyze_resume, generate_interview_question,
    evaluate_answer, extract_candidate_name, detect_language,
)

GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

st.set_page_config(page_title="Screen.ai",
                   page_icon="🔍", layout="wide",
                   initial_sidebar_state="collapsed")

# ── BASE CSS (only things Streamlit definitely applies) ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important}
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background:#080812!important;color:#f1f5f9!important}
.stApp{background:linear-gradient(160deg,#080812 0%,#0d0a1e 45%,#080f1e 100%)!important;min-height:100vh}
section[data-testid="stMain"] > div {padding-top:2rem!important}
.block-container{padding:0 2rem 2rem!important;max-width:100%!important}

/* Streamlit widget overrides */
div[data-testid="stFileUploader"]{background:#1e1e35!important;border:2px dashed rgba(99,102,241,.4)!important;border-radius:14px!important}
div[data-testid="stFileUploader"]:hover{border-color:#6366f1!important;background:#232340!important}
div[data-testid="stFileUploader"] *{color:#94a3b8!important}
div[data-testid="stFileUploader"] svg{fill:#6366f1!important}
div[data-testid="stFileUploader"] button{background:#2a2a4a!important;border:1px solid rgba(99,102,241,.4)!important;color:#c7d2fe!important;border-radius:8px!important}
textarea{background:#1e1e35!important;border:1px solid rgba(99,102,241,.35)!important;border-radius:12px!important;color:#f1f5f9!important;font-family:'Inter',sans-serif!important;caret-color:#f1f5f9!important}
textarea::placeholder{color:#4a5568!important}
div[data-baseweb="textarea"] > div{background:#1e1e35!important}
div[data-baseweb="textarea"] textarea{color:#f1f5f9!important;-webkit-text-fill-color:#f1f5f9!important}
.stButton>button{background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;color:#fff!important;border:none!important;border-radius:12px!important;padding:.72rem 1.5rem!important;font-weight:700!important;font-size:.95rem!important;width:100%!important;box-shadow:0 4px 20px rgba(99,102,241,.35)!important;transition:all .2s!important}
.stButton>button:hover{opacity:.9!important;transform:translateY(-2px)!important;box-shadow:0 8px 30px rgba(99,102,241,.5)!important}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(255,255,255,.05),rgba(255,255,255,.02))!important;border:1px solid rgba(255,255,255,.09)!important;border-radius:16px!important;padding:1.4rem!important}
[data-testid="stMetricValue"]{color:#f1f5f9!important;font-family:'JetBrains Mono'!important;font-size:2rem!important}
[data-testid="stMetricLabel"]{color:#64748b!important;font-size:.8rem!important}
.stExpander{background:rgba(255,255,255,.02)!important;border:1px solid rgba(255,255,255,.08)!important;border-radius:14px!important}
hr{border-color:rgba(255,255,255,.06)!important}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0d0d1a}::-webkit-scrollbar-thumb{background:rgba(99,102,241,.4);border-radius:3px}

/* ── RESPONSIVE MEDIA QUERIES ── */

/* Tablet (max 1024px) */
@media (max-width:1024px){
  .block-container{padding:0 1.2rem 1.5rem!important}
  [data-testid="stMetricValue"]{font-size:1.6rem!important}
}

/* Mobile (max 768px) */
@media (max-width:768px){
  .block-container{padding:0 .8rem 1rem!important}
  section[data-testid="stMain"] > div{padding-top:1rem!important}

  /* Stack columns vertically */
  [data-testid="column"]{width:100%!important;flex:100%!important;min-width:100%!important}
  [data-testid="stHorizontalBlock"]{flex-direction:column!important;gap:.8rem!important}

  /* Metrics smaller */
  [data-testid="stMetric"]{padding:.9rem!important;border-radius:12px!important}
  [data-testid="stMetricValue"]{font-size:1.5rem!important}
  [data-testid="stMetricLabel"]{font-size:.72rem!important}

  /* Textarea full width */
  textarea{font-size:.9rem!important}

  /* Button */
  .stButton>button{padding:.65rem 1rem!important;font-size:.88rem!important;border-radius:10px!important}

  /* Expander */
  .stExpander{border-radius:10px!important}
}

/* Small mobile (max 480px) */
@media (max-width:480px){
  .block-container{padding:0 .5rem .8rem!important}
  [data-testid="stMetricValue"]{font-size:1.3rem!important}
  .stButton>button{font-size:.82rem!important;padding:.6rem .8rem!important}
}
</style>
""", unsafe_allow_html=True)

# ── HERO via components.html (guaranteed to render) ─────────────────────────
components.html("""
<!DOCTYPE html>
<html>
<head>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{background:transparent;font-family:'Inter',sans-serif;padding:1.5rem 1rem 0}

@keyframes gradShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes floatY{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}

.hero{
  position:relative;overflow:hidden;text-align:center;
  padding:2.8rem 2rem 2.5rem;
  background:linear-gradient(135deg,#0c0928,#16115e,#0b2748,#080812);
  background-size:300% 300%;
  animation:gradShift 8s ease infinite;
  border:1px solid rgba(255,255,255,.1);
  border-radius:20px;
}
@media(max-width:768px){
  .hero{padding:2rem 1.2rem 1.8rem;border-radius:14px}
  .logo{font-size:2.8rem!important;letter-spacing:-2px!important}
  .sub{font-size:.9rem!important}
  .pills{gap:.4rem!important;margin-top:1.2rem!important}
  .pill{padding:.3rem .75rem!important;font-size:.72rem!important}
}
@media(max-width:480px){
  .hero{padding:1.5rem 1rem 1.5rem}
  .logo{font-size:2.2rem!important;letter-spacing:-1px!important}
  .sub{font-size:.82rem!important}
}
.hero::before{
  content:'';position:absolute;inset:0;
  background:
    radial-gradient(ellipse 60% 50% at 20% 60%,rgba(99,102,241,.22) 0%,transparent 70%),
    radial-gradient(ellipse 50% 40% at 80% 25%,rgba(6,182,212,.15) 0%,transparent 65%),
    radial-gradient(ellipse 40% 35% at 55% 85%,rgba(139,92,246,.12) 0%,transparent 60%);
  pointer-events:none;
}
.logo{
  font-size:3.8rem;font-weight:900;letter-spacing:-3px;line-height:1;
  background:linear-gradient(135deg,#818cf8 0%,#06b6d4 45%,#a78bfa 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  display:inline-block;
  animation:floatY 4s ease-in-out infinite, fadeUp .8s ease both;
  position:relative;
}
.sub{color:#94a3b8;font-size:1rem;margin-top:.5rem;font-weight:400;position:relative;
  animation:fadeUp .8s ease .15s both;}
.pills{display:flex;justify-content:center;flex-wrap:wrap;gap:.55rem;margin-top:1.5rem;position:relative;
  animation:fadeUp .8s ease .3s both;}
.pill{
  background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.15);
  border-radius:100px;padding:.35rem 1rem;font-size:.76rem;color:#c7d2fe;font-weight:500;
  backdrop-filter:blur(12px);transition:background .25s,transform .2s;cursor:default;
}
.pill:hover{background:rgba(99,102,241,.2);transform:translateY(-2px)}
@media(max-width:768px){
  body{padding:.8rem .6rem 0}
  .hero{padding:1.8rem 1rem 1.6rem;border-radius:14px}
  .logo{font-size:2.6rem;letter-spacing:-1.5px}
  .sub{font-size:.88rem}
  .pills{gap:.35rem;margin-top:1rem}
  .pill{padding:.28rem .7rem;font-size:.7rem}
}
@media(max-width:480px){
  body{padding:.5rem .4rem 0}
  .hero{padding:1.3rem .8rem 1.3rem}
  .logo{font-size:2rem;letter-spacing:-1px}
  .sub{font-size:.78rem}
}
</style>
</head>
<body>
<div class="hero">
  <div class="logo">Screen.ai</div>
  <div class="sub">AI-Powered Resume Screening &amp; Live Interview Platform</div>
  <div class="pills">
    <span class="pill">🧠 Smart Analysis</span>
    <span class="pill">📺 YouTube Learning</span>
    <span class="pill">🗺️ Skill Roadmap</span>
    <span class="pill">🎤 Live Interview</span>
  </div>
</div>
</body>
</html>
""", height=280)

# ── SECTION LABEL helper (renders properly in Streamlit) ────────────────────
def section_header(badge_text, badge_cls, title, subtitle):
    colors = {
        "indigo": ("#818cf8","rgba(99,102,241,.12)","rgba(99,102,241,.35)"),
        "violet": ("#a78bfa","rgba(139,92,246,.12)","rgba(139,92,246,.35)"),
        "cyan":   ("#22d3ee","rgba(6,182,212,.12)","rgba(6,182,212,.35)"),
        "green":  ("#34d399","rgba(16,185,129,.12)","rgba(16,185,129,.35)"),
        "pink":   ("#f472b6","rgba(236,72,153,.12)","rgba(236,72,153,.35)"),
        "amber":  ("#fcd34d","rgba(245,158,11,.12)","rgba(245,158,11,.35)"),
    }
    tc, bg, bd = colors.get(badge_cls, colors["indigo"])
    st.markdown(f"""
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,.4),rgba(6,182,212,.3),transparent);margin:2rem 0"></div>
    <div style="display:inline-flex;align-items:center;gap:.45rem;padding:.38rem 1rem;border-radius:100px;
                font-size:clamp(.62rem,.8vw,.71rem);font-weight:700;letter-spacing:.1em;text-transform:uppercase;
                margin-bottom:.75rem;background:{bg};border:1px solid {bd};color:{tc}">
      {badge_text}
    </div>
    <div style="font-size:clamp(1.15rem,2.5vw,1.55rem);font-weight:800;color:#f1f5f9;letter-spacing:-.03em;margin-bottom:.3rem">{title}</div>
    <div style="color:#64748b;font-size:clamp(.78rem,1.2vw,.875rem);line-height:1.65;margin-bottom:1.4rem">{subtitle}</div>
    """, unsafe_allow_html=True)

def divider():
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,.4),rgba(6,182,212,.3),transparent);margin:2rem 0"></div>', unsafe_allow_html=True)

def glass_card(content_html, extra_style=""):
    st.markdown(f"""
    <div style="background:linear-gradient(145deg,rgba(255,255,255,.045),rgba(255,255,255,.01));
                border:1px solid rgba(255,255,255,.08);border-radius:18px;
                padding:clamp(1rem,2vw,1.6rem);
                backdrop-filter:blur(16px);margin-bottom:1rem;{extra_style}">
      {content_html}
    </div>""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────────────────────────
def init_state():
    for k,v in {
        "analysis_done":False,"analysis_result":None,
        "cv_text":"","jd_text":"","candidate_name":"",
        "interview_started":False,"current_question":"",
        "q_audio_b64":"","interview_round":0,
        "answers_log":[],"evaluation":None,
        "q_history":[],"iv_lang":"english",
        "yt_cache":{},"yt_more":{},
    }.items():
        if k not in st.session_state: st.session_state[k]=v
init_state()

# ── HELPERS ──────────────────────────────────────────────────────────────────
def extract_pdf(f):
    r=PyPDF2.PdfReader(f)
    return "".join(p.extract_text() or "" for p in r.pages).strip()

def to_audio_b64(text,lang="en"):
    tts=gTTS(text=text,lang=lang,slow=False)
    buf=io.BytesIO(); tts.write_to_fp(buf); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def play_audio(b64, rnd=0):
    components.html(f"""<!DOCTYPE html><html><head>
    <style>
      body{{margin:0;padding:4px 0;background:transparent;font-family:Inter,sans-serif}}
      .pl{{display:flex;align-items:center;gap:.6rem;cursor:pointer;padding:.65rem 1rem;
           border-radius:10px;background:rgba(6,182,212,.1);border:1px solid rgba(6,182,212,.25);
           transition:background .2s;user-select:none}}
      .pl:hover{{background:rgba(6,182,212,.2)}}
      .tx{{font-size:.84rem;color:#67e8f9;font-weight:500}}
    </style></head><body>
    <audio id="au" preload="auto">
      <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    <div class="pl" id="btn" onclick="toggle()">
      <span style="font-size:1.2rem">🔊</span>
      <span class="tx" id="tx">AI is speaking...</span>
    </div>
    <script>
      var a=document.getElementById('au'),t=document.getElementById('tx');
      function toggle(){{if(a.paused){{a.currentTime=0;a.play();}}else{{a.pause();a.currentTime=0;}}}}
      a.onplay=function(){{t.textContent='🔈 Speaking... (click to stop)';}};
      a.onended=function(){{t.textContent='✅ Done — click to replay';}};
      a.onpause=function(){{t.textContent='▶️ Click to play';}};
      a.onerror=function(){{t.textContent='❌ Audio error — try again';}};
      a.play().catch(function(){{t.textContent='🔊 Click to hear AI voice';}});
    </script>
    </body></html>""", height=55)

def score_color(p): return "#10b981" if p>=75 else "#f59e0b" if p>=50 else "#ef4444"

def fetch_yt(skill,n=10):
    k=f"{skill}_{n}"
    if k in st.session_state.yt_cache: return st.session_state.yt_cache[k]
    try:
        d=requests.get("https://www.googleapis.com/youtube/v3/search",
            params={"part":"snippet","q":f"learn {skill} tutorial","type":"video","maxResults":n,"key":YOUTUBE_API_KEY},
            timeout=10).json()
        vids=[{"title":i["snippet"].get("title",""),"channel":i["snippet"].get("channelTitle",""),
               "thumb":i["snippet"].get("thumbnails",{}).get("medium",{}).get("url",""),
               "url":f"https://www.youtube.com/watch?v={i['id']['videoId']}"}
              for i in d.get("items",[]) if i.get("id",{}).get("videoId")]
        st.session_state.yt_cache[k]=vids; return vids
    except: return []

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — INPUT
# ═══════════════════════════════════════════════════════════════════════════════
section_header("⚡ Step 1 — Data Input","indigo","Upload Resume & Job Description",
               "Provide your CV and the job you are applying for.")

c1,c2=st.columns(2,gap="large")
with c1:
    st.markdown('<p style="font-weight:600;color:#e2e8f0;margin-bottom:.5rem">📄 Resume (PDF)</p>',unsafe_allow_html=True)
    pdf=st.file_uploader("pdf",type=["pdf"],label_visibility="collapsed")
    if pdf:
        st.success(f"✅ {pdf.name}")
        with st.spinner("Extracting..."):
            st.session_state.cv_text=extract_pdf(pdf)
        st.markdown(f"<div style='color:#475569;font-size:.8rem;margin-top:.3rem'>📊 {len(st.session_state.cv_text.split())} words extracted</div>",unsafe_allow_html=True)

with c2:
    st.markdown('<p style="font-weight:600;color:#e2e8f0;margin-bottom:.5rem">📝 Job Description</p>',unsafe_allow_html=True)
    jd=st.text_area("jd",height=200,
        placeholder="Paste the full job description here — include all requirements, responsibilities and skills...",
        label_visibility="collapsed")
    if jd: st.session_state.jd_text=jd

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════
section_header("🧠 Step 2 — AI Analysis","violet","Intelligent Resume Screening",
               "AI analyzes with implied skill intelligence — fair matching for every job type.")

# Show Analyze button only if CV and JD are provided and not yet analyzed
if not st.session_state.analysis_done:
    if st.session_state.cv_text and st.session_state.jd_text:
        if st.button("🚀 Analyze Resume", use_container_width=True, key="analyze_main"):
            if not GROQ_API_KEY:
                st.error("⚠️ GROQ_API_KEY missing in .env")
            else:
                with st.spinner("🔍 Analyzing with Llama 3.3 70B..."):
                    name=extract_candidate_name(st.session_state.cv_text,GROQ_API_KEY)
                    result=analyze_resume(st.session_state.cv_text,st.session_state.jd_text,GROQ_API_KEY)
                st.session_state.update({"candidate_name":name,"analysis_result":result,
                    "analysis_done":True,"interview_started":False,"interview_round":0,
                    "answers_log":[],"q_history":[],"yt_more":{},"evaluation":None})
                st.rerun()
    else:
        st.markdown('<div style="opacity:.4;pointer-events:none">', unsafe_allow_html=True)
        st.button("🚀 Analyze Resume", use_container_width=True, key="analyze_disabled", disabled=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#475569;font-size:.82rem;text-align:center;margin-top:.4rem">Upload a resume PDF and paste a job description to enable analysis</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.analysis_done and st.session_state.analysis_result:
    r=st.session_state.analysis_result
    pct=r["match_percentage"]; col=score_color(pct); name=st.session_state.candidate_name

    divider()

    # Candidate banner
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(99,102,241,.1),rgba(6,182,212,.07));
                border:1px solid rgba(99,102,241,.25);border-radius:14px;
                padding:1rem 1.2rem;margin-bottom:1.5rem;display:flex;align-items:center;
                gap:.8rem;flex-wrap:wrap">
      <span style="font-size:1.8rem">👤</span>
      <div style="flex:1;min-width:120px">
        <div style="font-weight:700;font-size:clamp(.9rem,2vw,1.1rem);color:#e2e8f0">{name}</div>
        <div style="font-size:.75rem;color:#475569;margin-top:.1rem">Candidate detected from CV</div>
      </div>
      <div style="background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.3);
                  border-radius:8px;padding:.28rem .7rem;font-size:.72rem;color:#818cf8;font-weight:700;
                  white-space:nowrap">
        AI Analyzed ✓
      </div>
    </div>""",unsafe_allow_html=True)

    # Score + metrics
    s1,s2,s3=st.columns([1.3,1,1],gap="medium")
    with s1:
        components.html(f"""
        <style>
        @keyframes pulseGlow{{0%,100%{{box-shadow:0 0 24px rgba(99,102,241,.2)}}50%{{box-shadow:0 0 52px rgba(99,102,241,.5)}}}}
        @keyframes scaleIn{{from{{opacity:0;transform:scale(.88)}}to{{opacity:1;transform:scale(1)}}}}
        body{{margin:0;font-family:'Inter',sans-serif;background:transparent}}
        .sc{{background:linear-gradient(145deg,#0f0c29,#16115e,#0b2540);
             border:1px solid rgba(99,102,241,.3);border-radius:20px;
             padding:2rem;text-align:center;position:relative;overflow:hidden;
             animation:pulseGlow 3.5s ease-in-out infinite,scaleIn .6s ease both}}
@media(max-width:768px){{
  .sc{{padding:1.3rem;border-radius:14px}}
  .num{{font-size:3.5rem;letter-spacing:-2px}}
  .lbl{{font-size:.65rem}}
}}
        .sc::before{{content:'';position:absolute;inset:0;
          background:radial-gradient(circle at 50% 0%,rgba(99,102,241,.18),transparent 65%);}}
        .num{{font-size:5rem;font-weight:900;line-height:1;
              font-family:'JetBrains Mono',monospace;letter-spacing:-4px;
              color:{col};position:relative}}
        .lbl{{font-size:.7rem;text-transform:uppercase;letter-spacing:.18em;
              color:#475569;margin-top:.5rem;position:relative}}
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@900&family=JetBrains+Mono:wght@600&display=swap" rel="stylesheet">
        <div class="sc"><div class="num">{pct}%</div><div class="lbl">Match Score</div></div>
        """,height=180)
    with s2:
        st.metric("✅ Matching Skills",len(r["matching_skills"]))
    with s3:
        st.metric("❌ Missing Skills",len(r["missing_skills"]))

    st.markdown("<br>",unsafe_allow_html=True)

    # Skills
    k1,k2=st.columns(2,gap="medium")
    with k1:
        st.markdown('<p style="font-weight:700;color:#e2e8f0;margin-bottom:.5rem">✅ Matching Skills</p>',unsafe_allow_html=True)
        chips="".join(f'<span style="display:inline-block;padding:.32rem .85rem;border-radius:8px;font-size:.77rem;font-weight:600;margin:.22rem;font-family:monospace;background:rgba(16,185,129,.11);border:1px solid rgba(16,185,129,.32);color:#34d399">{s}</span>' for s in r["matching_skills"]) or "<span style='color:#475569'>None found.</span>"
        glass_card(chips)
    with k2:
        st.markdown('<p style="font-weight:700;color:#e2e8f0;margin-bottom:.5rem">❌ Missing Skills</p>',unsafe_allow_html=True)
        chips="".join(f'<span style="display:inline-block;padding:.32rem .85rem;border-radius:8px;font-size:.77rem;font-weight:600;margin:.22rem;font-family:monospace;background:rgba(239,68,68,.11);border:1px solid rgba(239,68,68,.32);color:#f87171">{s}</span>' for s in r["missing_skills"]) or "<span style='color:#34d399;font-weight:700'>🎉 All skills matched!</span>"
        glass_card(chips)

    # Typing note
    if r.get("typing_suggestion"):
        st.markdown("""
        <div style="background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.28);
                    border-radius:12px;padding:1rem 1.25rem;margin:1rem 0;color:#fcd34d;
                    font-size:.875rem;display:flex;gap:.75rem;align-items:flex-start;line-height:1.6">
          <span style="font-size:1.3rem;flex-shrink:0">⌨️</span>
          <div><b style="color:#fbbf24">Typing Speed Note:</b> This role may require good typing speed.
          Verify with employer. Practice at
          <a href="https://www.typingtest.com" target="_blank" style="color:#fcd34d;text-decoration:underline">typingtest.com</a></div>
        </div>""",unsafe_allow_html=True)

    # Feedback
    st.markdown('<p style="font-weight:700;color:#e2e8f0;margin-bottom:.5rem">🤖 AI Feedback</p>',unsafe_allow_html=True)
    glass_card(f'<p style="color:#cbd5e1;font-size:.93rem;line-height:1.8;margin:0;border-left:4px solid {col};padding-left:1rem">{r["feedback"]}</p>')

    # ── ROADMAP ──────────────────────────────────────────────────────────────
    if r.get("roadmap"):
        section_header("🗺️ Skill Roadmap","cyan","Learning Path",
                       "Skills to develop in priority order for this role.")
        for i,item in enumerate(r["roadmap"]):
            p=item.get("priority","medium")
            pc={"high":"linear-gradient(135deg,#ef4444,#dc2626)",
                "medium":"linear-gradient(135deg,#f59e0b,#d97706)",
                "low":"linear-gradient(135deg,#10b981,#059669)"}.get(p,"linear-gradient(135deg,#f59e0b,#d97706)")
            pl={"high":"🔴 High","medium":"🟡 Medium","low":"🟢 Low"}.get(p,"Medium")
            st.markdown(f"""
            <div style="background:linear-gradient(145deg,rgba(255,255,255,.04),rgba(255,255,255,.01));
                        border:1px solid rgba(255,255,255,.08);border-radius:14px;
                        padding:clamp(.8rem,1.5vw,1.2rem) clamp(1rem,2vw,1.5rem);margin-bottom:.75rem;
                        display:flex;align-items:flex-start;gap:.8rem;flex-wrap:wrap;
                        transition:border-color .3s,transform .3s">
              <div style="width:36px;height:36px;border-radius:50%;background:{pc};
                          display:flex;align-items:center;justify-content:center;
                          font-weight:700;font-size:.85rem;flex-shrink:0;color:#fff">{item.get('step',i+1)}</div>
              <div style="flex:1;min-width:150px">
                <div style="font-weight:700;color:#e2e8f0;font-size:clamp(.85rem,1.5vw,.95rem)">{item.get('skill','')}</div>
                <div style="color:#64748b;font-size:clamp(.74rem,1.2vw,.81rem);margin-top:.2rem;line-height:1.5">{item.get('reason','')}</div>
              </div>
              <div style="font-size:.72rem;color:#475569;white-space:nowrap">{pl}</div>
            </div>""",unsafe_allow_html=True)

    # ── YOUTUBE ──────────────────────────────────────────────────────────────
    if r.get("missing_skills"):
        section_header("📺 YouTube Resources","pink","Learn Missing Skills",
                       "Curated videos — national & international creators.")
        for skill in r["missing_skills"]:
            with st.expander(f"📚  {skill}",expanded=False):
                mk=f"more_{skill}"
                n=10 if st.session_state.yt_more.get(mk) else 5
                with st.spinner(f"Loading videos for {skill}..."):
                    vids=fetch_yt(skill,10)
                if vids:
                    for idx,v in enumerate(vids[:n]):
                        if idx%3==0: cols=st.columns(3)
                        with cols[idx%3]:
                            st.markdown(f"""
                            <a href="{v['url']}" target="_blank" style="text-decoration:none">
                              <div style="background:linear-gradient(145deg,rgba(255,255,255,.05),rgba(255,255,255,.02));
                                          border:1px solid rgba(255,255,255,.08);border-radius:14px;
                                          overflow:hidden;margin-bottom:1rem;display:block;
                                          transition:border-color .3s,transform .3s">
                                <img src="{v['thumb']}" style="width:100%;aspect-ratio:16/9;object-fit:cover;display:block" alt="">
                                <div style="padding:.9rem">
                                  <div style="font-size:.82rem;font-weight:600;color:#e2e8f0;line-height:1.4;margin-bottom:.35rem">{v['title'][:68]}{'…' if len(v['title'])>68 else ''}</div>
                                  <div style="font-size:.72rem;color:#64748b">📺 {v['channel']}</div>
                                </div>
                              </div>
                            </a>""",unsafe_allow_html=True)
                    if len(vids)>5 and not st.session_state.yt_more.get(mk):
                        if st.button(f"▼ Show More Videos for {skill}",key=mk):
                            st.session_state.yt_more[mk]=True; st.rerun()
                else:
                    st.warning("Could not fetch videos. Check YouTube API key in .env")

    divider()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — INTERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.analysis_done:
    section_header("🎤 Step 3 — Live Interview","green","AI Mock Interview",
                   "Answer in Urdu or English — AI detects and responds in same language.")

    try:
        from streamlit_mic_recorder import mic_recorder; mic_ok=True
    except ImportError: mic_ok=False

    cand=st.session_state.candidate_name or "Candidate"

    if not st.session_state.interview_started:
        components.html(f"""
        <style>
        @keyframes scIn{{from{{opacity:0;transform:scale(.95)}}to{{opacity:1;transform:scale(1)}}}}
        @keyframes gradShift{{0%{{background-position:0% 50%}}50%{{background-position:100% 50%}}100%{{background-position:0% 50%}}}}
        body{{margin:0;font-family:'Inter',sans-serif;background:transparent;padding:4px 0}}
        .box{{
          background:linear-gradient(135deg,#0f0c29,#16115e,#0b2540);
          background-size:200% 200%;
          animation:gradShift 6s ease infinite, scIn .5s ease both;
          border:1px solid rgba(99,102,241,.35);border-radius:16px;
          padding:1.3rem 2rem;display:flex;align-items:center;gap:1rem;
          position:relative;overflow:hidden}}
@media(max-width:768px){{
  .box{{padding:1rem 1.2rem;gap:.7rem;border-radius:12px}}
  .ttl{{font-size:.95rem}}
  .em{{font-size:1.5rem}}
}}
@media(max-width:480px){{
  .box{{padding:.8rem 1rem}}
  .ttl{{font-size:.85rem}}
}}
        .box::before{{content:'';position:absolute;inset:0;
          background:radial-gradient(circle at 20% 50%,rgba(99,102,241,.15),transparent 60%);pointer-events:none}}
        .em{{font-size:1.8rem;flex-shrink:0;position:relative}}
        .ttl{{font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:0;position:relative}}
        .nm{{background:linear-gradient(135deg,#818cf8,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@700&display=swap" rel="stylesheet">
        <div class="box">
          <div class="em">🤖🎙️</div>
          <div class="ttl">AI Live Interview — <span class="nm">{cand}</span></div>
        </div>
        """,height=75)

        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("🎙️ Start Live Interview",use_container_width=True):
            with st.spinner("Preparing first question..."):
                res=st.session_state.analysis_result
                q=generate_interview_question(res["matching_skills"],res["missing_skills"],
                    GROQ_API_KEY,name=cand,language="english",
                    cv_text=st.session_state.cv_text,
                    jd_text=st.session_state.jd_text)
                a64=to_audio_b64(q,"en")
            st.session_state.update({"current_question":q,"q_audio_b64":a64,
                "interview_started":True,"evaluation":None,"q_history":[q]})
            st.session_state.interview_round+=1
            st.rerun()

    else:
        rnd=st.session_state.interview_round
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:.7rem 1rem;background:rgba(255,255,255,.02);
                    border:1px solid rgba(255,255,255,.08);border-radius:10px;margin-bottom:1rem">
          <span style="color:#64748b;font-size:.85rem">🎤 Round <b style="color:#e2e8f0">{rnd}</b></span>
          <span style="background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.3);
                       border-radius:100px;padding:.22rem .8rem;font-size:.72rem;
                       color:#34d399;font-weight:600">● Live</span>
        </div>
        <div style="background:rgba(6,182,212,.06);border-left:4px solid #06b6d4;
                    border-top:1px solid rgba(6,182,212,.15);border-right:1px solid rgba(6,182,212,.15);
                    border-bottom:1px solid rgba(6,182,212,.15);border-radius:0 14px 14px 0;
                    padding:clamp(.9rem,2vw,1.3rem) clamp(1rem,2vw,1.5rem);margin:1rem 0;color:#67e8f9;
                    font-size:clamp(.9rem,1.5vw,1.05rem);font-style:italic;line-height:1.7;word-break:break-word">
          💬 {st.session_state.current_question}
        </div>""",unsafe_allow_html=True)

        if st.session_state.q_audio_b64:
            play_audio(st.session_state.q_audio_b64,rnd)

        st.markdown("<br>",unsafe_allow_html=True)

        if mic_ok:
            st.markdown("**🎤 Record Your Answer:**")
            aud=mic_recorder(start_prompt="⏺️ Start",stop_prompt="⏹️ Stop",key=f"mic_{rnd}")
            if aud and aud.get("bytes"):
                st.success("✅ Recorded!")
                ans=f"[Voice — Round {rnd}]"
                with st.spinner("Evaluating..."):
                    ev=evaluate_answer(st.session_state.current_question,ans,GROQ_API_KEY,"english")
                st.session_state.evaluation=ev
                st.session_state.answers_log.append({"round":rnd,"question":st.session_state.current_question,"answer":ans,"evaluation":ev,"language":"english"})
        else:
            st.markdown("**✍️ Type Your Answer** *(Urdu ya English dono chalega)*")
            ans_txt=st.text_area("ans",placeholder="Yahan likhein / Type here...",
                                  key=f"a_{rnd}",label_visibility="collapsed",height=140)
            if st.button("📤 Submit Answer",key=f"s_{rnd}"):
                if ans_txt.strip():
                    with st.spinner("Evaluating..."):
                        lang=detect_language(ans_txt,GROQ_API_KEY)
                        ev=evaluate_answer(st.session_state.current_question,ans_txt,GROQ_API_KEY,lang)
                    st.session_state.evaluation=ev; st.session_state.iv_lang=lang
                    st.session_state.answers_log.append({"round":rnd,"question":st.session_state.current_question,
                        "answer":ans_txt,"evaluation":ev,"language":lang})
                    st.rerun()

        if st.session_state.evaluation:
            ev=st.session_state.evaluation; sc=ev.get("score",5)
            lang=st.session_state.iv_lang; sc_c=score_color(sc*10)
            st.markdown("<br>**📋 Evaluation**")
            e1,e2,e3=st.columns(3,gap="medium")
            with e1:
                components.html(f"""
                <style>
                @keyframes scIn{{from{{opacity:0;transform:scale(.9)}}to{{opacity:1;transform:scale(1)}}}}
                body{{margin:0;font-family:'Inter',sans-serif;background:transparent}}
                .ec{{background:linear-gradient(145deg,rgba(255,255,255,.04),rgba(255,255,255,.01));
                     border:1px solid rgba(255,255,255,.08);border-radius:14px;
                     padding:1.2rem;text-align:center;animation:scIn .5s ease both}}
                .sc{{font-size:3rem;font-weight:900;color:{sc_c};font-family:'JetBrains Mono',monospace;letter-spacing:-2px}}
                .sl{{font-size:.7rem;color:#475569;text-transform:uppercase;letter-spacing:.12em;margin-top:.3rem}}
                </style>
                <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@600&display=swap" rel="stylesheet">
                <div class="ec"><div class="sc">{sc}<span style="font-size:1.3rem;color:#334155">/10</span></div>
                <div class="sl">Your Score</div></div>
                """,height=120)
            with e2:
                lbl="Strengths" if lang=="english" else "Achhi Baat"
                glass_card(f'<div style="font-weight:700;color:#34d399;margin-bottom:.5rem;font-size:.84rem">✅ {lbl}</div><div style="color:#94a3b8;font-size:.83rem;line-height:1.6">{ev.get("strengths","")}</div>')
            with e3:
                lbl2="Improve" if lang=="english" else "Behtar Karein"
                glass_card(f'<div style="font-weight:700;color:#fbbf24;margin-bottom:.5rem;font-size:.84rem">💡 {lbl2}</div><div style="color:#94a3b8;font-size:.83rem;line-height:1.6">{ev.get("improvement","")}</div>')

            st.markdown("<br>",unsafe_allow_html=True)
            b1,b2=st.columns(2,gap="medium")
            with b1:
                if st.button("▶️ Next Question"):
                    res=st.session_state.analysis_result
                    # Always use the DETECTED language from candidate's last answer
                    lang=st.session_state.iv_lang
                    with st.spinner("Generating next question..."):
                        # Always generate fresh question in detected language
                        # Do NOT use follow_up from eval — it may be in wrong language
                        q=generate_interview_question(
                            res["matching_skills"],res["missing_skills"],GROQ_API_KEY,
                            name=cand,language=lang,
                            conversation_history=st.session_state.q_history,
                            cv_text=st.session_state.cv_text,
                            jd_text=st.session_state.jd_text)
                        tl="ur" if lang=="urdu" else "en"
                        try: a64=to_audio_b64(q,tl)
                        except: a64=to_audio_b64(q,"en")
                    st.session_state.update({"current_question":q,"q_audio_b64":a64,"evaluation":None})
                    st.session_state.interview_round+=1
                    st.session_state.q_history.append(q)
                    st.rerun()
            with b2:
                if st.button("🏁 End Interview"):
                    st.session_state.interview_started=False; st.rerun()

        if st.session_state.answers_log:
            with st.expander(f"📜 Interview History — {len(st.session_state.answers_log)} rounds"):
                for e in st.session_state.answers_log:
                    fl="🇵🇰" if e.get("language")=="urdu" else "🇬🇧"
                    st.markdown(f"**Round {e['round']}** {fl}")
                    st.markdown(f"<span style='color:#64748b;font-size:.84rem'>Q: {e['question']}</span>",unsafe_allow_html=True)
                    if e.get("evaluation"): st.markdown(f"Score: **{e['evaluation'].get('score','?')}/10**")
                    st.markdown("---")

# ── ANALYZE ANOTHER RESUME — only after interview section ───────────────────
if st.session_state.analysis_done:
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,.4),rgba(6,182,212,.3),transparent);margin:2.5rem 0 1.5rem"></div>',unsafe_allow_html=True)
    if st.button("🔄 Analyze Another Resume",use_container_width=True,key="analyze_another"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:3rem 0 2rem;margin-top:2rem;border-top:1px solid rgba(255,255,255,.05)">
  <div style="font-size:1.5rem;font-weight:900;letter-spacing:-1px;
              background:linear-gradient(135deg,#818cf8,#06b6d4);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text;margin-bottom:.4rem">Screen.ai</div>
  <div style="font-size:.77rem;color:#334155">
    Powered by <span style="color:#818cf8">Groq Llama 3.3 70B</span> ·
    <span style="color:#22d3ee">YouTube Data API v3</span> · v2.0
  </div>
</div>""",unsafe_allow_html=True)
