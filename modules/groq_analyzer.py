"""
modules/groq_analyzer.py
Screen.ai — AI Career Portal
Author: Muhammad Shamikh | Full Stack Developer
"""

import json
import re
from groq import Groq


def extract_candidate_name(cv_text: str, api_key: str) -> str:
    client = Groq(api_key=api_key)
    prompt = f"""Extract the full name of the candidate from this CV/Resume.
Return ONLY the full name — nothing else. No explanation, no punctuation.
If not found, return "Candidate".

CV:
{cv_text[:800]}"""
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        name = r.choices[0].message.content.strip().split("\n")[0].strip()
        return name if name else "Candidate"
    except Exception:
        return "Candidate"


def analyze_resume(cv_text: str, jd_text: str, api_key: str) -> dict:
    client = Groq(api_key=api_key)

    system_prompt = """You are an expert ATS (Applicant Tracking System) analyst with deep understanding of implied and transferable skills.

CORE PHILOSOPHY:
- Skills are often IMPLIED by other skills. You must recognize these relationships.
- A candidate with relevant experience should NEVER get 0% match if they have transferable skills.

IMPLIED SKILLS RULES (very important):
- MS Office / Microsoft Office → implies: Data Entry, Basic Computer Skills, Word Processing, Spreadsheet Management
- Excel → implies: Data Entry, Data Analysis basics, Spreadsheet skills
- Any office software → implies: Typing ability (do NOT list typing as missing skill EVER)
- Programming in any language → implies: Problem Solving, Logical Thinking
- Customer Service experience → implies: Communication, Email Writing
- Any accounting software → implies: Data Entry, Excel basics
- Project management experience → implies: Team Leadership, Planning
- Teaching/Training experience → implies: Communication, Presentation skills

TYPING SPEED RULE — CRITICAL:
- NEVER add "Typing Speed", "Typing", "WPM", or any typing-related skill to missing_skills list
- Instead, add a note in feedback: "Company may want to verify typing speed before hiring"
- If JD mentions typing speed, acknowledge it in feedback only

MATCHING RULES:
- Be generous with matching — if candidate has implied/related skills, count them as matching
- match_percentage should reflect REAL employability, not just keyword matching
- A candidate with MS Office skills applying for Data Entry should get at least 60-70% match

STRICT OUTPUT — return ONLY this JSON, no markdown, no explanation:
{
  "match_percentage": <integer 0-100>,
  "matching_skills": [<explicitly matching + implied matching skills>],
  "missing_skills": [<only skills EXPLICITLY in JD that candidate truly lacks — NO typing>],
  "feedback": "<2-3 sentences about fit, mention typing test suggestion if relevant>",
  "typing_suggestion": <true/false — true if job needs typing>,
  "roadmap": [
    {"step": 1, "skill": "<skill name>", "reason": "<why learn this first>", "priority": "high/medium/low"},
    ...
  ]
}"""

    user_prompt = f"""RESUME/CV:
---
{cv_text[:4000]}
---

JOB DESCRIPTION:
---
{jd_text[:2000]}
---

Analyze thoroughly considering implied and transferable skills. Return strict JSON."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        result = json.loads(raw)

        # Safety: remove any typing-related items from missing_skills
        typing_keywords = ["typing", "wpm", "words per minute", "typing speed", "type"]
        missing = [
            s for s in result.get("missing_skills", [])
            if not any(kw in s.lower() for kw in typing_keywords)
        ]

        return {
            "match_percentage": int(result.get("match_percentage", 0)),
            "matching_skills": list(result.get("matching_skills", [])),
            "missing_skills": missing,
            "feedback": str(result.get("feedback", "No feedback available.")),
            "typing_suggestion": bool(result.get("typing_suggestion", False)),
            "roadmap": list(result.get("roadmap", [])),
        }

    except json.JSONDecodeError as e:
        return {
            "match_percentage": 0,
            "matching_skills": [],
            "missing_skills": [],
            "feedback": f"⚠️ Parsing error: {str(e)}",
            "typing_suggestion": False,
            "roadmap": [],
        }
    except Exception as e:
        return {
            "match_percentage": 0,
            "matching_skills": [],
            "missing_skills": [],
            "feedback": f"⚠️ Analysis failed: {str(e)}",
            "typing_suggestion": False,
            "roadmap": [],
        }


def detect_language(text: str, api_key: str) -> str:
    """Detect if text is Urdu or English."""
    client = Groq(api_key=api_key)
    prompt = f"""What language is this text written in? Reply with ONLY one word: "urdu" or "english".
Text: {text[:200]}"""
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        lang = r.choices[0].message.content.strip().lower()
        return "urdu" if "urdu" in lang else "english"
    except Exception:
        return "english"


def generate_interview_question(
    matching_skills: list,
    missing_skills: list,
    api_key: str,
    name: str = "Candidate",
    language: str = "english",
    conversation_history: list = None,
    cv_text: str = "",
    jd_text: str = "",
) -> str:
    client = Groq(api_key=api_key)

    # Build rich context from CV and JD
    skills_context = ""
    if matching_skills:
        skills_context += f"Candidate HAS these skills (from CV): {', '.join(matching_skills[:8])}.\n"
    if missing_skills:
        skills_context += f"Candidate is MISSING these JD skills: {', '.join(missing_skills[:5])}.\n"

    cv_snippet = f"CV Summary:\n{cv_text[:800]}\n" if cv_text else ""
    jd_snippet = f"Job Description Summary:\n{jd_text[:600]}\n" if jd_text else ""

    history_context = ""
    if conversation_history:
        history_context = "Previously asked questions (DO NOT repeat):\n" + "\n".join(
            [f"- {q}" for q in conversation_history[-5:]]
        )

    if language == "urdu":
        lang_instruction = "MANDATORY: Write the question in Roman Urdu ONLY. No English words except technical terms."
    else:
        lang_instruction = "Write the question in English ONLY."

    prompt = f"""You are a strict professional HR interviewer. Your job is to ask ONLY questions directly relevant to the job description and candidate's CV.

CANDIDATE NAME: {name}

{cv_snippet}
{jd_snippet}
{skills_context}
{history_context}

STRICT RULES — MUST FOLLOW:
1. Question MUST be directly related to the Job Description requirements
2. Question MUST be based on candidate's actual skills or experience from CV
3. Ask about specific tools, software, tasks mentioned in the JD (e.g. MS Word, Excel, Photoshop, data entry, etc.)
4. Do NOT ask generic questions like "tell me about yourself" or "where do you see yourself in 5 years"
5. Do NOT ask about skills not mentioned in JD or CV
6. Start with candidate's name: "{name},"
7. Keep it 1-2 sentences only
8. Be open-ended (not yes/no)
9. {lang_instruction}

Return ONLY the interview question. Nothing else."""

    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=150,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        if language == "urdu":
            return f"{name}, aap {matching_skills[0] if matching_skills else 'apne kaam'} ke baare mein batayein?"
        if matching_skills:
            return f"{name}, can you describe how you have used {matching_skills[0]} in your previous work?"
        return f"{name}, what specific tasks from this job description have you performed before?"


def evaluate_answer(question: str, answer: str, api_key: str, language: str = "english") -> dict:
    client = Groq(api_key=api_key)

    if language == "urdu":
        lang_instruction = "Respond ONLY in Urdu or Roman Urdu. All fields must be in Urdu."
        follow_lang = "Urdu ya Roman Urdu mein agla sawaal poochein."
    else:
        lang_instruction = "Respond in English."
        follow_lang = "Ask the follow-up question in English."

    prompt = f"""You are a STRICT professional interviewer evaluating a candidate's answer.

Question asked: {question}
Candidate's answer: {answer}

STRICT SCORING RULES:
- Score 1-3: Answer is irrelevant, off-topic, nonsensical, or does not address the question at all
- Score 4-5: Answer is vague, incomplete, or only partially relevant
- Score 6-7: Answer is relevant but lacks depth or specific examples
- Score 8-9: Answer is strong, relevant, detailed with good examples
- Score 10: Perfect, exceptional answer

IMPORTANT: If the answer is irrelevant, random text, or does not answer the question — give score 1-3 maximum. Do NOT be generous with scores.

{lang_instruction}

Return ONLY this JSON (no markdown, no extra text):
{{
  "score": <integer 1-10>,
  "strengths": "<one sentence about what was good, or say nothing was good if irrelevant>",
  "improvement": "<one sentence on improvement>",
  "follow_up": "<one follow-up question — {follow_lang}>"
}}"""

    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        raw = r.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception:
        if language == "urdu":
            return {
                "score": 3,
                "strengths": "Jawab mila.",
                "improvement": "Sawaal se mutalliq jawab dein.",
                "follow_up": "Kya aap dobara koshish kar sakte hain?",
            }
        return {
            "score": 3,
            "strengths": "A response was provided.",
            "improvement": "Please answer the question directly.",
            "follow_up": "Can you try answering the question again?",
        }
