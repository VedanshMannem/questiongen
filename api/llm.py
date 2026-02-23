import json
import os
import re
from typing import Any, Dict, List, Optional, cast
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DEFAULT_MODEL = "gemini-3-flash-preview"
DEBUG_LOGS = os.getenv("SATQ_DEBUG", "1").lower() not in {"0", "false", "no"}
DEBUG_VERBOSE = os.getenv("SATQ_DEBUG_VERBOSE", "0").lower() in {"1", "true", "yes"}

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def _extract_text_from_response(resp: Any) -> str:
    if resp is None:
        return ""

    if hasattr(resp, "text") and getattr(resp, "text"):
        return str(getattr(resp, "text"))

    if isinstance(resp, dict):
        for key in ("text", "output", "result"):
            if key in resp and resp[key]:
                return str(resp[key])

        candidates = resp.get("candidates") or resp.get("outputs") or []
        if candidates:
            first = candidates[0]
            if isinstance(first, dict):
                content = first.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, dict):
                    parts = content.get("parts") or []
                    texts = [p.get("text") for p in parts if isinstance(p, dict) and p.get("text")]
                    if texts:
                        return "\n".join(str(x) for x in texts)
            return str(first)

    if hasattr(resp, "candidates"):
        candidates = getattr(resp, "candidates")
        if candidates:
            first = candidates[0]
            content = getattr(first, "content", None)
            if content is not None:
                parts = getattr(content, "parts", None)
                if parts:
                    texts = []
                    for part in parts:
                        txt = getattr(part, "text", None)
                        if txt:
                            texts.append(str(txt))
                    if texts:
                        return "\n".join(texts)
                return str(content)
            return str(first)

    return str(resp)

def _invoke_model(prompt: str) -> str:
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    is_generation_prompt = (
        "Generate one SAT Reading and Writing" in prompt
        or "Generate one SAT-style question" in prompt
    )

    if DEBUG_LOGS and (DEBUG_VERBOSE or is_generation_prompt):
        preview = prompt[:700].replace("\n", "\\n")
        print(f"[DEBUG][llm] invoking Gemini model={DEFAULT_MODEL} prompt_len={len(prompt)}")
        print(f"[DEBUG][llm] prompt_preview={preview}")

    response = client.models.generate_content(model=DEFAULT_MODEL, contents=prompt)
    text = _extract_text_from_response(response)
    if DEBUG_LOGS and (DEBUG_VERBOSE or is_generation_prompt):
        resp_preview = (text or "")[:700].replace("\n", "\\n")
        print(f"[DEBUG][llm] response_len={len(text or '')}")
        print(f"[DEBUG][llm] response_preview={resp_preview}")
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


def _parse_json_with_recovery(text: str) -> Any:
    cleaned = _clean_json_text(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    array_match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            pass

    obj_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if obj_match:
        return json.loads(obj_match.group(0))

    raise ValueError("No valid JSON object/array found in model output")

def extract_graph_nodes(
    questions_batch: List[Dict[str, Any]],
    skill_name: str,
) -> List[Dict[str, Any]]:

    
    # Build ultra-compact representation - minimal tokens
    minified_qs = []
    for q in questions_batch:
        minified_qs.append({
            "q": q.get("question_text", "")[:50],  # Further reduce to 50 chars
            "ans": q.get("correct_answer", "")[:20],  # Truncate answer too
        })
    
    prompt = f"""You are extracting SAT item patterns for skill {skill_name}.
For each question, return one object with keys: topic, logic_skeleton, answer_skeleton.
Keep values short and reusable.
Questions: {json.dumps(minified_qs, separators=(',', ':'))}
Return JSON array only."""

    raw_text = _invoke_model(prompt)
    try:
        result = _parse_json_with_recovery(raw_text)
        
        return result
    except Exception as e:
        if DEBUG_LOGS:
            print(f"[DEBUG][llm] extract_graph_nodes parse_error={e}")
            print(f"[DEBUG][llm] raw_output={raw_text}")
        text = _clean_json_text(raw_text)
        raise RuntimeError(f"Failed to parse JSON from LLM output: {e}\nOutput was:\n{text}") from e

def generate_question(
    topic: str,
    logic_skeleton: str,
    answer_skeleton: str,
    example_question: Dict[str, Any],
    dataset_context: Optional[Dict[str, Any]] = None,
    user_context: Optional[str] = None,
) -> Dict[str, Any]:

    ex_len = max(120, len(example_question.get("prompt", "")))

    prompt = f"""Generate one SAT Reading and Writing multiple-choice item.
        Constraints:
        - Topic: {topic}
        - Reasoning skeleton: {logic_skeleton}
        - Answer pattern: {answer_skeleton}
        - Similar prompt length: about {ex_len} characters
        - Exactly 4 answer choices total (1 correct + 3 distractors)
        - Must be an original scenario.
        - Do not copy, closely paraphrase, or reuse named entities from the sample.

        Example style reference:
        {json.dumps(example_question, ensure_ascii=False)}

        Dataset context (metadata/length/topic/skeleton priors):
        {json.dumps(dataset_context or {}, ensure_ascii=False)}

        Optional user context:
        {user_context or ""}

        Return JSON only:
        {{
            "prompt": "...",
            "question_text": "...",
            "correct_answer_text": "...",
            "distractors": ["...", "...", "..."],
            "explanation": "..."
        }}"""

    raw_text = _invoke_model(prompt)
    try:
        return cast(Dict[str, Any], _parse_json_with_recovery(raw_text))
    except Exception as e:
        if DEBUG_LOGS:
            print(f"[DEBUG][llm] generate_question parse_error={e}")
            print(f"[DEBUG][llm] raw_output={raw_text}")
        text = _clean_json_text(raw_text)
        raise RuntimeError(f"Failed to parse JSON from LLM output: {e}\nOutput was:\n{text}") from e


def generate_question_from_features(
    question_topic: str,
    question_skeleton: str,
    answer_type: str,
    sample_question: Dict[str, Any],
    style_profile: Dict[str, Any],
    dataset_context: Optional[Dict[str, Any]] = None,
    user_context: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = f"""Generate one SAT-style question.
Inputs:
- topic: {question_topic}
- skeleton: {question_skeleton}
- answer_type: {answer_type}
- style_profile: {json.dumps(style_profile, ensure_ascii=False)}
- dataset_context: {json.dumps(dataset_context or {}, ensure_ascii=False)}
- sample_question: {json.dumps(sample_question, ensure_ascii=False)}
- user_context: {user_context or ''}

Hard requirements:
- Generate a fresh scenario each time.
- Do not copy, near-copy, or lightly paraphrase the sample question text.
- Do not reuse key named entities from the sample.

If answer_type is multiple_choice:
Return JSON with keys prompt, question_text, correct_answer_text, distractors (exactly 3), explanation.

If answer_type is free_response:
Return JSON with keys prompt, question_text, expected_answer, explanation.

Return JSON only."""

    raw_text = _invoke_model(prompt)
    try:
        return cast(Dict[str, Any], _parse_json_with_recovery(raw_text))
    except Exception as e:
        if DEBUG_LOGS:
            print(f"[DEBUG][llm] generate_question_from_features parse_error={e}")
            print(f"[DEBUG][llm] raw_output={raw_text}")
        text = _clean_json_text(raw_text)
        raise RuntimeError(f"Failed to parse JSON from LLM output: {e}\nOutput was:\n{text}") from e


def generate_question_set_from_plan(
    plan: List[Dict[str, Any]],
    sample_question: Dict[str, Any],
    dataset_context: Optional[Dict[str, Any]] = None,
    user_context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    prompt = f"""Generate a SAT-style question set in one response.
Inputs:
- generation_plan: {json.dumps(plan, ensure_ascii=False)}
- sample_question: {json.dumps(sample_question, ensure_ascii=False)}
- dataset_context: {json.dumps(dataset_context or {}, ensure_ascii=False)}
- user_context: {user_context or ''}

Rules:
- Return exactly one question object per generation_plan item, in the same order.
- Keep each item original and distinct.
- Do not copy or near-copy sample text.
- For multiple_choice items: include exactly 1 correct_answer_text and exactly 3 distractors.
- For free_response items: include expected_answer.

Return JSON array only, where each item has:
{{
  "prompt": "...",
  "question_text": "...",
  "answer_type": "multiple_choice|free_response",
  "correct_answer_text": "...",
  "distractors": ["...", "...", "..."],
  "expected_answer": "...",
  "explanation": "..."
}}"""

    raw_text = _invoke_model(prompt)
    try:
        parsed = _parse_json_with_recovery(raw_text)
        if not isinstance(parsed, list):
            raise ValueError("Model output for question set must be a JSON array")
        return cast(List[Dict[str, Any]], parsed)
    except Exception as e:
        if DEBUG_LOGS:
            print(f"[DEBUG][llm] generate_question_set_from_plan parse_error={e}")
            print(f"[DEBUG][llm] raw_output={raw_text}")
        text = _clean_json_text(raw_text)
        raise RuntimeError(f"Failed to parse question set JSON: {e}\nOutput was:\n{text}") from e
