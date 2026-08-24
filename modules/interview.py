"""Live interview engine: question generation, transcription, scoring, report."""
from modules import config
from modules.groq_client import GroqError, chat_json, transcribe

_SYSTEM = (
    "You are a senior interviewer conducting a fair mock interview for any "
    "profession. You are bilingual (English and Urdu/Roman-Urdu) and always "
    "respond strictly in valid JSON."
)


def generate_questions(cv_text, jd_text, matching_skills, missing_skills,
                       n=config.INTERVIEW_QUESTIONS):
    """Generate personalised interview questions from CV + JD context."""
    cv_text = (cv_text or "")[: config.MAX_INPUT_CHARS]
    jd_text = (jd_text or "")[: config.MAX_INPUT_CHARS]

    prompt = f"""Create exactly {n} interview questions for this candidate.

RULES:
- Base questions ONLY on the CV and JD context below — no generic filler.
- Progress from a warm-up to deeper, role-specific and scenario questions.
- Each question must be a single, clear sentence.
- Keep them answerable by voice in under a minute.

Matching skills: {", ".join(matching_skills) or "n/a"}
Skills to probe (gaps): {", ".join(missing_skills) or "n/a"}

CV:
\"\"\"{cv_text}\"\"\"

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

Return ONLY: {{"questions": ["q1", "q2", ...]}} with exactly {n} questions."""

    try:
        result = chat_json(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=1600,
        )
    except GroqError:
        result = {}

    questions = [str(q).strip() for q in (result.get("questions") or []) if str(q).strip()]
    if not questions:
        # Safe fallback so the interview can always start.
        skill = matching_skills[0] if matching_skills else "your field"
        questions = [
            f"Tell me about your experience with {skill}.",
            "Describe a challenging problem you solved and how you approached it.",
            "How do you keep your skills up to date?",
            "Why are you a good fit for this role?",
        ]
    return questions[:n]


def transcribe_answer(audio_bytes, fmt="webm"):
    """Transcribe recorded audio to text. Returns (text, error)."""
    ext = (fmt or "webm").lower().lstrip(".")
    try:
        return transcribe(audio_bytes, filename=f"answer.{ext}"), None
    except GroqError as exc:
        return "", str(exc)


def evaluate_answer(question, answer_text, jd_text):
    """Score a single interview answer. Bilingual, honest scoring."""
    answer_text = (answer_text or "").strip()
    if not answer_text:
        return {
            "score": 1,
            "language": "English",
            "strengths": "—",
            "improvements": "No answer was provided. Please respond to the question.",
        }

    prompt = f"""Evaluate the candidate's answer honestly and fairly.

RULES:
- Detect the answer's language (English or Urdu/Roman-Urdu) and write your
  feedback in that SAME language.
- Score 1-10. Irrelevant, empty or off-topic answers get 1-3 honestly.
- Be specific and constructive.

QUESTION: {question}
JOB CONTEXT: {(jd_text or "")[:1500]}
CANDIDATE ANSWER: {answer_text}

Return ONLY:
{{"score": 0, "language": "English|Urdu", "strengths": "...", "improvements": "..."}}"""

    try:
        result = chat_json(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
    except GroqError as exc:
        return {
            "score": 0,
            "language": "English",
            "strengths": "—",
            "improvements": f"Could not evaluate this answer: {exc}",
        }

    return {
        "score": _clamp_score(result.get("score")),
        "language": (result.get("language") or "English").strip(),
        "strengths": (result.get("strengths") or "—").strip(),
        "improvements": (result.get("improvements") or "—").strip(),
    }


def generate_final_report(answers, match_percentage, jd_text):
    """Summarise the whole interview into a hire recommendation."""
    transcript = "\n\n".join(
        f"Q{i + 1}: {a['question']}\nAnswer: {a.get('answer', '')}\n"
        f"Score: {a.get('eval', {}).get('score', 0)}/10"
        for i, a in enumerate(answers)
    )

    prompt = f"""Summarise this mock interview into a final report.

CV/JD match score was {match_percentage}%.
Job context: {(jd_text or "")[:1500]}

INTERVIEW TRANSCRIPT:
{transcript}

Return ONLY:
{{
  "overall_score": 0,               // 0-100
  "recommendation": "Strong Hire | Hire | Maybe | Not Recommended",
  "summary": "2-3 sentence overall verdict",
  "strengths": ["..."],
  "improvements": ["..."]
}}"""

    try:
        result = chat_json(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1800,
        )
    except GroqError as exc:
        return {
            "overall_score": _avg_score(answers),
            "recommendation": "Maybe",
            "summary": f"Automated summary unavailable ({exc}). Score based on answer averages.",
            "strengths": [],
            "improvements": [],
        }

    return {
        "overall_score": _clamp_pct(result.get("overall_score"), _avg_score(answers)),
        "recommendation": (result.get("recommendation") or "Maybe").strip(),
        "summary": (result.get("summary") or "").strip(),
        "strengths": [str(s).strip() for s in (result.get("strengths") or []) if str(s).strip()],
        "improvements": [str(s).strip() for s in (result.get("improvements") or []) if str(s).strip()],
    }


def _clamp_score(value):
    try:
        return max(1, min(10, round(float(value))))
    except (TypeError, ValueError):
        return 1


def _clamp_pct(value, default=0):
    try:
        return max(0, min(100, round(float(value))))
    except (TypeError, ValueError):
        return default


def _avg_score(answers):
    scores = [a.get("eval", {}).get("score", 0) for a in answers]
    return round(sum(scores) / len(scores) * 10) if scores else 0
