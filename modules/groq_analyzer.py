"""CV vs job description analysis engine.

Match percentage is computed in Python from a per-skill present/absent
list rather than a single number the model invents, so the score on
screen always matches the skills shown and stays stable across reruns.
"""
import re

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
2. List EVERY distinct skill/requirement the JD explicitly or implicitly
   asks for as one entry in required_skills, each marked present=true if it
   (directly or by clear implication) exists in the CV, else present=false.
3. Mark a skill present=false ONLY if it is genuinely absent (directly and by
   implication) from the CV. Do not be overly harsh, give credit for
   transferable/implied experience per rule 1.
4. NEVER list "typing" as a required/missing skill. If relevant, mention it
   only as a gentle tip inside feedback.
5. Do not fabricate skills the CV does not support.
6. For each present=false skill, add a concise, priority-ordered learning
   step to roadmap.
7. Keep feedback constructive, specific and under 60 words.
8. Be consistent: given the same CV and JD, your classification of each
   skill as present/absent should not change between runs.
"""

_SCHEMA = """
Return ONLY this JSON structure (no extra keys, no commentary):
{
  "candidate_name": "Best guess of the candidate's full name from the CV, or 'Candidate'",
  "required_skills": [
    {"skill": "name of one required/implied skill from the JD", "present": true}
  ],
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
    cv_text = config.smart_truncate(cv_text or "")
    jd_text = config.smart_truncate(jd_text or "")

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
            # temperature=0.0 for maximum run-to-run consistency.
            temperature=0.0,
            max_tokens=2800,
        )
    except GroqError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"Unexpected error during analysis: {exc}"}

    if not isinstance(result, dict):
        return {"error": "Received an unexpected response format from the model."}

    matching_skills, missing_skills, match_percentage = _score_from_required_skills(
        result.get("required_skills")
    )
    # Reconcile the two lists against the CV text before returning.
    matching_skills, missing_skills = _reconcile(matching_skills, missing_skills, cv_text)
    # Recompute the percentage from the reconciled lists so the score
    # always matches the skills shown on screen.
    total = len(matching_skills) + len(missing_skills)
    if total:
        match_percentage = round(len(matching_skills) / total * 100)

    # Normalise / harden the payload so the UI can rely on every field.
    return {
        "candidate_name": (result.get("candidate_name") or "Candidate").strip(),
        "match_percentage": _clamp_pct(match_percentage),
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "roadmap": _as_roadmap(result.get("roadmap")),
        "feedback": (result.get("feedback") or "").strip(),
    }


def _score_from_required_skills(required_skills):
    """Turn the model's per-skill present/absent list into matching_skills,
    missing_skills and match_percentage, all computed from the same list
    so the percentage can never contradict the skills shown.

    Falls back to (empty, empty, 50) if the model didn't return a usable
    list.
    """
    if not isinstance(required_skills, list) or not required_skills:
        return [], [], 50

    matching, missing = [], []
    for item in required_skills:
        if isinstance(item, dict):
            name = str(item.get("skill") or "").strip()
            present = bool(item.get("present"))
        elif isinstance(item, str):
            # No present/absent flag, treat as missing so it still surfaces.
            name, present = item.strip(), False
        else:
            continue
        if not name:
            continue
        (matching if present else missing).append(name)

    total = len(matching) + len(missing)
    pct = round(len(matching) / total * 100) if total else 50
    return matching, missing, pct


def _normalize_skill(skill):
    """Lowercase, strip punctuation/whitespace, collapse internal spaces,
    so 'React.js', 'React JS' and 'react' are recognised as the same skill."""
    s = re.sub(r"[^a-z0-9+#. ]", "", str(skill).lower().strip())
    return re.sub(r"\s+", " ", s).strip()


def _mentioned_in_text(skill_norm, haystack_norm):
    """Containment check with light suffix tolerance, e.g. 'testing'
    matches a resume that says 'tested'."""
    if not skill_norm:
        return False
    if skill_norm in haystack_norm:
        return True
    for suffix in ("ing", "ed", "s"):
        if skill_norm.endswith(suffix) and len(skill_norm) > len(suffix) + 2:
            stem = skill_norm[: -len(suffix)]
            if stem and stem in haystack_norm:
                return True
    return False


def _reconcile(matching_raw, missing_raw, cv_text):
    """De-dupe both lists and move any "missing" skill that's actually
    present in the CV text into matching_skills instead.
    """
    cv_norm = re.sub(r"[^a-z0-9+#. ]", "", (cv_text or "").lower())
    cv_norm = re.sub(r"\s+", " ", cv_norm)

    seen = {}
    matching = []
    for skill in matching_raw:
        norm = _normalize_skill(skill)
        if norm and norm not in seen:
            seen[norm] = True
            matching.append(skill)

    missing = []
    for skill in missing_raw:
        norm = _normalize_skill(skill)
        if not norm or norm in seen:
            continue
        if _mentioned_in_text(norm, cv_norm):
            seen[norm] = True
            matching.append(skill)
        else:
            seen[norm] = True
            missing.append(skill)

    return matching, missing


def _clamp_pct(value):
    try:
        return max(0, min(100, round(float(value))))
    except (TypeError, ValueError):
        return 0


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
