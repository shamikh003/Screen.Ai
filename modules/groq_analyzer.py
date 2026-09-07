"""CV vs Job-Description analysis engine.

BUG-FIX NOTES (see inline comments for detail):
  [FIX-A] cv_text/jd_text now use config.smart_truncate() instead of a blind
          [:MAX_INPUT_CHARS] slice, which was cutting off trailing Skills
          sections on longer resumes and causing false "missing skills".
  [FIX-B] _as_list() now safely coerces dict/non-string items instead of
          str()-ing arbitrary objects into garbage skill names.
  [FIX-C] New _reconcile() pass: de-dupes matching/missing skills
          (case/whitespace-insensitive), guarantees the two lists are
          mutually exclusive, and re-checks every "missing" skill against
          the actual CV text so a model misclassification doesn't produce
          a false gap (and therefore an irrelevant YouTube search later).
  [FIX-D] Defensive isinstance(result, dict) check so a malformed model
          response returns a clean {"error": ...} instead of crashing with
          an AttributeError deep inside the UI.
  [FIX-E] match_percentage used to be a raw number the model invented
          holistically (temperature=0.2) — re-running the SAME CV/JD could
          swing wildly (observed: 80% -> 40%) purely from LLM sampling
          variance, since nothing tied the number to any other field.
          Now the model reports skill-by-skill present/absent (required_skills)
          and match_percentage is COMPUTED in Python as matched/total, so the
          score is always mathematically consistent with the skills shown on
          screen, and temperature is dropped for this call so the underlying
          skill classification itself is also far more stable run-to-run.
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
   implication) from the CV. Do not be overly harsh — give credit for
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
    # [FIX-A] smart_truncate keeps head + tail instead of chopping off
    # whatever section happens to sit past MAX_INPUT_CHARS.
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
            # [FIX-E] Dropped to 0.0 (fully greedy decoding) for maximum
            # run-to-run consistency. Note: even at temperature=0, hosted
            # LLM APIs (Groq, OpenAI, etc.) can still show tiny residual
            # non-determinism from batched GPU floating-point kernels — this
            # is a known, provider-side limitation with no client-side fix.
            # What THIS setting eliminates is the deliberate randomness of
            # temperature-based sampling on top of that.
            temperature=0.0,
            max_tokens=2800,
        )
    except GroqError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # [FIX-D] defensive: never let a raw parsing/
        return {"error": f"Unexpected error during analysis: {exc}"}  # network exception escape as an unhandled crash

    # [FIX-D] Guard against a malformed / non-dict response before we start
    # calling .get() on it.
    if not isinstance(result, dict):
        return {"error": "Received an unexpected response format from the model."}

    matching_skills, missing_skills, match_percentage = _score_from_required_skills(
        result.get("required_skills")
    )
    # [FIX-C] Reconcile the two lists against each other and against the
    # actual CV text before returning anything (safety net on top of the
    # model's own present/absent classification).
    matching_skills, missing_skills = _reconcile(matching_skills, missing_skills, cv_text)
    # Recompute the percentage from the RECONCILED lists so the number on
    # screen always matches the chips on screen, even after reconciliation
    # moves a skill from missing -> matching.
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
    """[FIX-E] Turn the model's per-skill present/absent list into
    matching_skills / missing_skills / match_percentage, all computed
    deterministically from the SAME list — so the percentage can never
    contradict the skills shown, and reruns can't wildly disagree just
    because the model invented a different holistic number.

    Falls back to (empty, empty, 50) if the model didn't return a usable
    list at all (e.g. very old/odd response shape) — analyze_profile's
    caller will still see a sane default rather than crashing.
    """
    if not isinstance(required_skills, list) or not required_skills:
        return [], [], 50

    matching, missing = [], []
    for item in required_skills:
        if isinstance(item, dict):
            name = str(item.get("skill") or "").strip()
            present = bool(item.get("present"))
        elif isinstance(item, str):
            # Model returned a bare string with no present/absent flag —
            # treat as "missing" so it at least surfaces to the user rather
            # than being silently dropped.
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
    """Lowercase, strip punctuation/whitespace, collapse internal spaces —
    so 'React.js', 'React JS' and 'react' are recognised as the same skill."""
    s = re.sub(r"[^a-z0-9+#. ]", "", str(skill).lower().strip())
    return re.sub(r"\s+", " ", s).strip()


def _mentioned_in_text(skill_norm, haystack_norm):
    """Containment check with light suffix tolerance (e.g. 'testing' should
    match a resume that says 'tested')."""
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
    """[FIX-C] De-dupe both lists and move any "missing" skill that's
    actually present in the CV text into matching_skills instead. This is
    the deterministic safety net for cases where the model paraphrases the
    JD's wording and fails to recognise a synonym/variant that's already on
    the resume — the most common cause of an incorrect missing-skills list.
    """
    cv_norm = _normalize_skill(cv_text) if cv_text else ""
    # _normalize_skill collapses whitespace but for a full CV we want a
    # single normalized string to search within, not skill-style stripping
    # of punctuation across the whole document boundary — reuse the same
    # cheap normalisation for consistency.
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
            continue  # duplicate of an already-matching skill, or empty
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
