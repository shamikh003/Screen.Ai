"""CV vs Job-Description analysis engine."""
from modules import config
from modules.groq_client import GroqError, chat_json

_SYSTEM = (
    "You are an experienced, fair technical recruiter. You evaluate any job "
    "in any field (tech, design, medical, finance, data entry, teaching, "
    "trades, etc.). You always respond strictly in valid JSON."
)

_RULES = """
EVALUATION RULES:
1. Understand IMPLIED skills. Example: 'MS Office' implies data entry & word
   processing; 'React' implies JavaScript & HTML/CSS; a medical degree implies
   patient care. Credit transferable and implied skills fairly.
2. NEVER give an unfair 0%. If the candidate has any relevant or transferable
   background, reflect it in the score. Only near-0 for a total mismatch.
3. Mark a skill 'missing' ONLY if it is explicitly required by the JD and is
   genuinely absent (directly and by implication) from the CV.
4. NEVER list "typing" as a missing skill. If relevant, mention it only as a
   gentle tip inside feedback.
5. Do not fabricate skills the CV does not support.
6. For each missing skill, add a concise, priority-ordered learning step.
7. Keep feedback constructive, specific and under 60 words.
"""

_SCHEMA = """
Return ONLY this JSON structure (no extra keys, no commentary):
{
  "candidate_name": "Best guess of the candidate's full name from the CV, or 'Candidate'",
  "match_percentage": 0,
  "matching_skills": ["skills present in CV that the JD wants"],
  "missing_skills": ["skills explicitly required by JD but absent"],
  "roadmap": [
    {"skill": "name", "priority": "High | Medium | Low", "action": "one concrete learning step"}
  ],
  "feedback": "constructive summary for the candidate"
}
"""


def analyze_profile(cv_text, jd_text):
    """Analyze a CV against a JD.

    Returns a dict with the analysis, or ``{"error": "..."}`` on failure.
    """
    cv_text = (cv_text or "")[: config.MAX_INPUT_CHARS]
    jd_text = (jd_text or "")[: config.MAX_INPUT_CHARS]

    prompt = f"""{_RULES}

CV TEXT:
\"\"\"{cv_text}\"\"\"

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

{_SCHEMA}"""

    try:
        result = chat_json(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2800,
        )
    except GroqError as exc:
        return {"error": str(exc)}

    # Normalise / harden the payload so the UI can rely on every field.
    return {
        "candidate_name": (result.get("candidate_name") or "Candidate").strip(),
        "match_percentage": _clamp_pct(result.get("match_percentage")),
        "matching_skills": _as_list(result.get("matching_skills")),
        "missing_skills": _as_list(result.get("missing_skills")),
        "roadmap": _as_roadmap(result.get("roadmap")),
        "feedback": (result.get("feedback") or "").strip(),
    }


def _clamp_pct(value):
    try:
        return max(0, min(100, round(float(value))))
    except (TypeError, ValueError):
        return 0


def _as_list(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _as_roadmap(value):
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        if isinstance(item, dict) and item.get("skill"):
            cleaned.append(
                {
                    "skill": str(item.get("skill")).strip(),
                    "priority": str(item.get("priority") or "Medium").strip().title(),
                    "action": str(item.get("action") or "").strip(),
                }
            )
    return cleaned
