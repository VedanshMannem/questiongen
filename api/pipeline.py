import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from . import llm
from . import supermemory_client
from .storage import create_run_dir, save_json

REQUIRED_FIELDS = ["prompt", "question_text", "answer_choices"]

class PipelineError(Exception):
    pass

def _load_questions(upload_path: str) -> List[Dict[str, Any]]:
    with open(upload_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]

    if not isinstance(data, list):
        raise PipelineError("Input JSON must be an array of questions or an object with a 'questions' array.")

    return data

def _normalize_question(q: Dict[str, Any], index: int) -> Dict[str, Any]:
    normalized = {
        "id": q.get("id") or f"q_{index + 1}",
        "prompt": str(q.get("prompt", "")),
        "question_text": str(q.get("question_text", "")),
        "answer_choices": q.get("answer_choices") or [],
        "correct_answer": q.get("correct_answer"),
        "explanation": str(q.get("explanation", "")),
        "metadata": q.get("metadata") or {},
    }

    if not isinstance(normalized["answer_choices"], list):
        normalized["answer_choices"] = []

    return normalized

def _validate_questions(questions: List[Dict[str, Any]]) -> None:
    for i, q in enumerate(questions):
        for field in REQUIRED_FIELDS:
            if field not in q or not q[field]:
                raise PipelineError(f"Question {i + 1} missing required field: {field}")

        if not isinstance(q.get("answer_choices"), list) or len(q["answer_choices"]) < 2:
            raise PipelineError(f"Question {i + 1} must include at least 2 answer choices.")

def _stats(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    return {
        "average": round(sum(values) / len(values), 2),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }

def _compute_insights(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall_q_len = []
    overall_a_len = []
    overall_expl_len = []

    by_domain = defaultdict(list)
    by_skill = defaultdict(list)
    by_difficulty = defaultdict(list)
    by_detailed = defaultdict(list)

    for q in questions:
        prompt = q.get("prompt") or ""
        q_text = q.get("question_text") or ""
        answers = q.get("answer_choices") or []
        explanation = q.get("explanation") or ""

        q_length = len(prompt) + len(q_text)
        avg_ans_len = sum(len(a) for a in answers) / len(answers) if answers else 0
        expl_len = len(explanation)

        overall_q_len.append(q_length)
        overall_a_len.append(avg_ans_len)
        overall_expl_len.append(expl_len)

        meta = q.get("metadata") or {}
        domain = (meta.get("domain") or "Unknown").strip()
        skill = (meta.get("skill") or "Unknown").strip()
        difficulty = (meta.get("difficulty") or "Unknown").strip()

        stats_obj = {"q_len": q_length, "a_len": avg_ans_len}
        by_domain[domain].append(stats_obj)
        by_skill[(domain, skill)].append(stats_obj)
        by_difficulty[(domain, difficulty)].append(stats_obj)
        by_detailed[(domain, skill, difficulty)].append(stats_obj)

    def group_stats(group_data: List[Dict[str, float]]) -> Dict[str, Any]:
        q_lens = [x["q_len"] for x in group_data]
        a_lens = [x["a_len"] for x in group_data]
        return {
            "question_length": _stats(q_lens),
            "answer_length": _stats(a_lens),
        }

    insights = {
        "overall": {
            "question_length": _stats(overall_q_len),
            "answer_length": _stats(overall_a_len),
            "explanation_length": _stats(overall_expl_len),
        },
        "by_domain": [],
        "by_skill": [],
        "by_difficulty": [],
        "detailed": [],
    }

    for domain in sorted(by_domain.keys()):
        insights["by_domain"].append({
            "domain": domain,
            "stats": group_stats(by_domain[domain])
        })

    for key in sorted(by_difficulty.keys()):
        domain, difficulty = key
        insights["by_difficulty"].append({
            "domain": domain,
            "difficulty": difficulty,
            "stats": group_stats(by_difficulty[key])
        })

    for key in sorted(by_skill.keys()):
        domain, skill = key
        insights["by_skill"].append({
            "domain": domain,
            "skill": skill,
            "stats": group_stats(by_skill[key])
        })

    for key in sorted(by_detailed.keys()):
        domain, skill, difficulty = key
        insights["detailed"].append({
            "domain": domain,
            "skill": skill,
            "difficulty": difficulty,
            "stats": group_stats(by_detailed[key])
        })

    return insights

def _summarize_for_memory(
    insights: Dict[str, Any],
    catalogs: Optional[Dict[str, List[str]]] = None,
) -> str:
    overall = insights.get("overall") or {}
    q_len = overall.get("question_length") or {}
    a_len = overall.get("answer_length") or {}

    lines = [
        "Dataset summary:",
        f"- Avg question length: {q_len.get('average')} (min {q_len.get('min')}, max {q_len.get('max')})",
        f"- Avg answer length: {a_len.get('average')} (min {a_len.get('min')}, max {a_len.get('max')})",
    ]

    if catalogs:
        topics = catalogs.get("topics") or []
        skills = catalogs.get("logic_skeletons") or []
        answers = catalogs.get("answer_skeletons") or []

        if topics:
            lines.append("Top topics (sample): " + ", ".join(topics[:10]))
        if skills:
            lines.append("Logic skeletons (sample): " + ", ".join(skills[:8]))
        if answers:
            lines.append("Answer skeletons (sample): " + ", ".join(answers[:8]))

    return "\n".join(lines)

def _group_by_skill(questions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for q in questions:
        skill = q.get("metadata", {}).get("skill") or "Unknown"
        grouped[skill].append(q)
    return grouped

def _extract_nodes(questions: List[Dict[str, Any]], batch_size: int = 3) -> List[Dict[str, Any]]:
    """Extract nodes with optimized batching.
    
    Uses smaller batch sizes (3 instead of 5) to:
    - Reduce API load per request
    - Prevent timeouts on large batches
    - Get faster responses
    
    Also deduplicates to avoid redundant API calls.
    """
    grouped = _group_by_skill(questions)
    all_nodes: List[Dict[str, Any]] = []
    seen_patterns: set = set()  # Track seen (topic, logic, answer) to avoid redundant extractions

    for skill, skill_questions in grouped.items():
        for i in range(0, len(skill_questions), batch_size):
            batch = skill_questions[i:i + batch_size]
            minified = []
            for q in batch:
                minified.append({
                    "id": q.get("id"),
                    "prompt": q.get("prompt"),
                    "question_text": q.get("question_text"),
                    "correct_answer": q.get("correct_answer"),
                })
            
            nodes = llm.extract_graph_nodes(minified, skill)
            for node in nodes:
                node_id = node.get("id")
                if not node_id and minified:
                    node_id = minified[0].get("id")
                node["id"] = node_id
                
                # Deduplication: skip if we've already extracted this pattern
                pattern_key = (
                    node.get("topic", ""),
                    node.get("logic_skeleton", ""),
                    node.get("answer_skeleton", "")
                )
                if pattern_key not in seen_patterns:
                    all_nodes.append(node)
                    seen_patterns.add(pattern_key)

    return all_nodes

def _build_catalogs(nodes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    topics = set()
    logic_skeletons = set()
    answer_skeletons = set()

    for node in nodes:
        topic = node.get("Topic") or node.get("topic") or ""
        logic = node.get("Logic Skeleton") or node.get("logic_skeleton") or ""
        answer = node.get("Answer Skeleton") or node.get("answer_skeleton") or ""
        if topic:
            topics.add(topic)
        if logic:
            logic_skeletons.add(logic)
        if answer:
            answer_skeletons.add(answer)

    return {
        "topics": sorted(topics),
        "logic_skeletons": sorted(logic_skeletons),
        "answer_skeletons": sorted(answer_skeletons),
    }

def _pick_example(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "prompt": questions[0].get("prompt"),
        "question_text": questions[0].get("question_text"),
        "answer_choices": questions[0].get("answer_choices"),
        "correct_answer": questions[0].get("correct_answer"),
        "explanation": questions[0].get("explanation"),
    }

def _finalize_generation(raw: Dict[str, Any]) -> Dict[str, Any]:
    correct_text = raw["correct_answer_text"]
    distractors = raw["distractors"]

    all_choices = list(distractors) + [correct_text]
    random.shuffle(all_choices)

    correct_index = all_choices.index(correct_text)
    correct_letter = ["A", "B", "C", "D"][correct_index]

    return {
        "prompt": raw["prompt"],
        "question_text": raw["question_text"],
        "answer_choices": all_choices,
        "correct_answer": correct_letter,
        "explanation": raw["explanation"],
    }

def _generate_questions(
    catalogs: Dict[str, List[str]],
    example_question: Dict[str, Any],
    count: int,
    user_context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    topics = catalogs.get("topics") or []
    logic_skeletons = catalogs.get("logic_skeletons") or []
    answer_skeletons = catalogs.get("answer_skeletons") or []

    if not topics or not logic_skeletons or not answer_skeletons:
        raise PipelineError("Insufficient extracted nodes to generate questions.")

    generated = []
    for _ in range(count):
        topic = random.choice(topics)
        logic = random.choice(logic_skeletons)
        answer = random.choice(answer_skeletons)
        raw = llm.generate_question(topic, logic, answer, example_question, user_context=user_context)
        generated.append(_finalize_generation(raw))

    return generated

def run_pipeline(
    upload_path: str,
    run_id: str,
    extract_nodes: bool,
    generate_count: int,
    user_id: Optional[str] = None,
    memory_query: Optional[str] = None,
) -> Dict[str, Any]:
    run_dir = create_run_dir(run_id)
    questions_raw = _load_questions(upload_path)
    normalized = [_normalize_question(q, i) for i, q in enumerate(questions_raw)]

    _validate_questions(normalized)
    save_json(os.path.join(run_dir, "input_normalized.json"), normalized)

    insights = _compute_insights(normalized)
    save_json(os.path.join(run_dir, "insights.json"), insights)

    nodes: List[Dict[str, Any]] = []
    catalogs: Dict[str, List[str]] = {}
    generated: List[Dict[str, Any]] = []

    if extract_nodes:
        nodes = _extract_nodes(normalized)
        save_json(os.path.join(run_dir, "graph_nodes.json"), nodes)
        catalogs = _build_catalogs(nodes)
        save_json(os.path.join(run_dir, "catalogs.json"), catalogs)

    if user_id:
        memory_payload = _summarize_for_memory(insights, catalogs if catalogs else None)
        supermemory_client.store_user_memory(user_id, memory_payload)

    user_context = ""
    if user_id and generate_count > 0:
        query = memory_query or "User preferences for question topics, style, difficulty, and answer constraints."
        user_context = supermemory_client.build_user_context(user_id, query)

    if generate_count > 0:
        if not catalogs:
            catalogs = _build_catalogs(nodes)
        example = _pick_example(normalized)
        generated = _generate_questions(catalogs, example, generate_count, user_context=user_context)
        save_json(os.path.join(run_dir, "generated_questions.json"), generated)

    summary = {
        "run_id": run_id,
        "input_count": len(normalized),
        "extract_nodes": extract_nodes,
        "generate_count": generate_count,
        "user_id": user_id,
        "artifacts": {
            "input_normalized": "input_normalized.json",
            "insights": "insights.json",
            "graph_nodes": "graph_nodes.json" if extract_nodes else None,
            "catalogs": "catalogs.json" if extract_nodes else None,
            "generated_questions": "generated_questions.json" if generate_count > 0 else None,
        },
    }

    save_json(os.path.join(run_dir, "run.json"), summary)
    return summary
