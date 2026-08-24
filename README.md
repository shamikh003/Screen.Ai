# 🔍 Screen.ai — AI Career Portal

<div align="center">

![Screen.ai](https://img.shields.io/badge/Screen.ai-AI%20Career%20Portal-6366f1?style=for-the-badge&logo=streamlit&logoColor=white)

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-screen--ai.streamlit.app-06b6d4?style=for-the-badge)](https://screen-ai.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-gpt--oss%20%2B%20Whisper-f97316?style=for-the-badge)](https://groq.com)

**AI-Powered Resume Screening & Live Voice Interview Platform**
*Works for ANY job — Web Dev, Data Entry, Design, Medical, Finance, Engineering & more*

[🌐 Live Demo](https://screen-ai.streamlit.app) · [👨‍💻 Author](https://linkedin.com/in/muhammad-shamikh) · [⭐ Star this repo](#)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Universal Job Analysis** | Works for any job — just paste the JD and let AI do the rest |
| 🎯 **Smart Match Scoring** | Implied skill intelligence — MS Office = Data Entry understood |
| 📊 **Fair 0% Protection** | Never unfairly scores 0% for relevant backgrounds |
| 🗺️ **Learning Roadmap** | Priority-ordered skill development plan for missing skills |
| 📺 **YouTube Resources** | Auto-fetches learning videos for each missing skill |
| 🎤 **Live AI Interview** | Voice-based mock interview with personalized JD-based questions |
| 🌐 **Bilingual** | Auto-detects Urdu/English — responds in candidate's language |
| ⌨️ **Typing Smart** | Never lists typing as missing skill — gives it as a suggestion |
| 📱 **Responsive** | Works on desktop, tablet, and mobile |

---

## 🌍 Works For Any Job

```
💻 Web Developer      → HTML, CSS, React, Node.js
📊 Data Analyst       → Excel, Python, SQL, Power BI
🎨 Graphic Designer   → Photoshop, Illustrator, Figma
📢 Digital Marketing  → SEO, Google Ads, Social Media
💰 Accountant         → QuickBooks, Excel, Taxation
🔧 Mechanical Eng     → AutoCAD, SolidWorks, CAD/CAM
📱 App Developer      → Flutter, React Native, Swift
🤖 AI Engineer        → Python, TensorFlow, PyTorch
🏥 Medical/Doctor     → MBBS, Specialization, Experience
👨‍🏫 Teacher           → Subject Knowledge, Communication
📝 Data Entry         → MS Office, Excel, Typing
... and any other job — just paste the JD!
```

---

## 🚀 Tech Stack

```
Frontend    →  Streamlit (Python)
LLM         →  Groq API — gpt-oss (ultra-fast JSON-mode inference)
Speech→Text →  Groq Whisper (whisper-large-v3-turbo)
TTS         →  gTTS — Google Text-to-Speech
PDF Parser  →  PyPDF2
Videos      →  YouTube Data API v3
UI          →  Custom dark glassmorphism CSS
```

---

## 🖥️ How It Works

```
1. 📄 Upload PDF Resume
         ↓
2. 📝 Paste Job Description (any job, any field)
         ↓
3. 🤖 AI extracts candidate name from CV
         ↓
4. 🧠 Groq LLM analyzes CV vs JD
    → Implied skills understood
    → Fair match percentage
    → Missing skills identified
         ↓
5. 📊 Results Dashboard
    → Match Score (%)
    → Matching & Missing Skills
    → Learning Roadmap
    → YouTube Videos per skill
         ↓
6. 🎤 Live AI Voice Interview
    → Questions based on CV + JD
    → Urdu or English auto-detected
         ↓
7. 📋 Real-time Evaluation
    → Score (1-10)
    → Strengths & Improvements
    → Next question generated
```

---

## ⚙️ Local Setup

### 1. Clone
```bash
git clone https://github.com/shamikh003/Screen.Ai.git
cd Screen.Ai
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Create `.env`
```env
GROQ_API_KEY=your_groq_key_here
YOUTUBE_API_KEY=your_youtube_key_here
```

### 4. Run
```bash
streamlit run app.py
```

---

## 🔑 API Keys (Both Free)

| API | Get It Here | Free Limit |
|---|---|---|
| Groq API | [console.groq.com](https://console.groq.com) | Generous free tier |
| YouTube Data API v3 | [console.cloud.google.com](https://console.cloud.google.com) | 10,000 req/day |

---

## 📁 Project Structure

```
Screen.Ai/
├── app.py                  # Main Streamlit app (UI + flow)
├── requirements.txt        # Dependencies
├── .env.example            # Template for your API keys
├── .streamlit/
│   └── config.toml         # Theme configuration
├── modules/
│   ├── __init__.py
│   ├── config.py           # Central config & secret loading
│   ├── groq_client.py      # Groq REST client (chat + Whisper)
│   ├── groq_analyzer.py    # CV vs JD analysis engine
│   ├── interview.py        # Question gen, transcription, scoring
│   └── youtube.py          # Learning-video lookup
└── README.md
```

---

## 🧠 AI Intelligence Highlights

- **Implied Skills** — Has MS Office? → Data Entry, Word Processing implied ✅
- **No Unfair 0%** — Transferable skills always considered ✅
- **Typing Protection** — Never listed as missing skill, shown as tip only ✅
- **JD-Strict Interview** — Questions only from CV + JD context ✅
- **Strict Scoring** — Irrelevant answers scored 1-3/10 honestly ✅
- **Language Detection** — Urdu jawab → Urdu question, English → English ✅

---

## 👨‍💻 Author

**Muhammad Shamikh** — Full Stack Developer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Muhammad%20Shamikh-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/muhammad-shamikh)
[![GitHub](https://img.shields.io/badge/GitHub-shamikh003-181717?style=for-the-badge&logo=github)](https://github.com/shamikh003)

---

## 📄 License

MIT License — free to use, modify and distribute.

---

<div align="center">
  <b>⭐ If this helped you, please star the repo!</b><br><br>
  <i>Built with ❤️ by Muhammad Shamikh</i>
</div>
