import json
import os
import time
import importlib
from typing import Any, Dict, List, Optional, cast
from google import genai  

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

model_name = "gemini-2.5-pro"

# Rate limiting and caching configuration
MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("GEMINI_MIN_SECONDS", "1.0"))  # Flash Lite is faster and cheaper
REQUEST_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "30.0"))  # Timeout to prevent hanging
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))  # Retry failed requests

_last_call_time = 0.0
_client_instance: Optional[Any] = None
_node_cache: Dict[str, List[Dict[str, Any]]] = {}  # Cache extracted nodes by skill

def _pace_calls() -> None:
    """Rate limiter to respect API quotas and prevent timeouts"""
    global _last_call_time
    now = time.time()
    delta = now - _last_call_time
    if delta < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - delta)
    _last_call_time = time.time()

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
    
    for attempt in range(MAX_RETRIES):
        try:
            _pace_calls()  # Rate limit before each attempt
            
            # Pattern 1: genai.GenerativeModel(...).generate_content(prompt) - Primary method
            if hasattr(genai, "GenerativeModel"):
                ModelCls = getattr(genai, "GenerativeModel")
                resp = client.models.generate_content(
                    model=model_name, contents=prompt
                )
                text = _extract_text_from_response(resp)
                if text:
                    return text
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"API request failed after {MAX_RETRIES} attempts: {e}") from e
            time.sleep(1)  # Brief wait before retry
            continue

    raise RuntimeError("All API attempts exhausted")

def extract_graph_nodes(
    questions_batch: List[Dict[str, Any]],
    skill_name: str,
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    
    cache_key = f"{skill_name}:{len(questions_batch)}"
    if cache_key in _node_cache:
        return _node_cache[cache_key]
    
    # Build ultra-compact representation - minimal tokens
    minified_qs = []
    for q in questions_batch:
        minified_qs.append({
            "q": q.get("question_text", "")[:50],  # Further reduce to 50 chars
            "ans": q.get("correct_answer", "")[:20],  # Truncate answer too
        })
    
    prompt = f"""Extract 3 patterns per question from {len(questions_batch)} {skill_name} Qs.
Patterns: topic, logic_skeleton, answer_skeleton.
{json.dumps(minified_qs, separators=(',', ':'))}
JSON only: [{{"topic":"...","logic_skeleton":"...","answer_skeleton":"..."}}]"""

    raw_text = _invoke_model(prompt)
    text = _clean_json_text(raw_text)
    try:
        result = json.loads(text)
        _node_cache[cache_key] = result  # Cache the results
        return result
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from LLM output: {e}\nOutput was:\n{text}") from e

def generate_question(
    topic: str,
    logic_skeleton: str,
    answer_skeleton: str,
    example_question: Dict[str, Any],
    user_context: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:

    # Extract minimal style hints
    ex_len = max(100, len(example_question.get("prompt", "")))
    ex_ans_len = 20  # Typical SAT answer length
    
    # Compact prompt - minimal verbosity
    prompt = f"""Gen SAT q: {topic}|{logic_skeleton}|{answer_skeleton}
Passage:{ex_len}c,answer:{ex_ans_len}c,3distractors
JSON:{{"prompt":"...","question_text":"...","correct_answer_text":"...","distractors":[...],"explanation":"..."}}"""

    raw_text = _invoke_model(prompt)
    text = _clean_json_text(raw_text)
    try:
        return json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from LLM output: {e}\nOutput was:\n{text}") from e
