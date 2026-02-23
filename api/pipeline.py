import json
import os
import random
import re
import hashlib
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional

from . import llm
from . import dataset_analysis
from .storage import create_run_dir, data_dir, run_path, save_json

REQUIRED_FIELDS = ["prompt", "question_text"]


class PipelineError(Exception):
    pass


DEBUG_LOGS = os.getenv("SATQ_DEBUG", "1").lower() not in {"0", "false", "no"}
LLM_EXTRACTION_MAX_QUESTIONS = max(1, int(os.getenv("SATQ_LLM_EXTRACTION_MAX_QUESTIONS", "20")))
LLM_EXTRACTION_BATCH_SIZE = max(1, min(20, int(os.getenv("SATQ_LLM_EXTRACTION_BATCH_SIZE", "20"))))
CACHE_VERSION = "v3"
CACHE_ARTIFACTS = [
    "input_normalized.json",
    "insights.json",
    "question_features.json",
    "graph_nodes.json",
    "catalogs.json",
    "dataset_analysis.json",
    "features.json",
]


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

        if "answer_choices" in q and q["answer_choices"]:
            if not isinstance(q["answer_choices"], list) or len(q["answer_choices"]) < 2:
                raise PipelineError(f"Question {i + 1} answer_choices must be a list with at least 2 items.")


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
        insights["by_domain"].append({"domain": domain, "stats": group_stats(by_domain[domain])})

    for key in sorted(by_difficulty.keys()):
        domain, difficulty = key
        insights["by_difficulty"].append({
            "domain": domain,
            "difficulty": difficulty,
            "stats": group_stats(by_difficulty[key]),
        })

    for key in sorted(by_skill.keys()):
        domain, skill = key
        insights["by_skill"].append({"domain": domain, "skill": skill, "stats": group_stats(by_skill[key])})

    for key in sorted(by_detailed.keys()):
        domain, skill, difficulty = key
        insights["detailed"].append({
            "domain": domain,
            "skill": skill,
            "difficulty": difficulty,
            "stats": group_stats(by_detailed[key]),
        })

    return insights


def _infer_answer_type(q: Dict[str, Any]) -> str:
    choices = q.get("answer_choices") or []
    if isinstance(choices, list) and len(choices) >= 2:
        return "multiple_choice"
    return "free_response"


def _infer_question_skeleton(prompt: str, question_text: str) -> str:
    _, _, template = dataset_analysis.infer_skeleton_template({"prompt": prompt, "question_text": question_text})
    return template


def _clean_text_snippet(text: str, max_len: int) -> str:
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    return compact[:max_len]


def _infer_topic(q: Dict[str, Any]) -> str:
    metadata = q.get("metadata") or {}
    domain = (metadata.get("domain") or "").strip()
    skill = (metadata.get("skill") or "").strip()
    section = (metadata.get("section") or "").strip()

    if domain and skill:
        return f"{domain} / {skill}"
    if domain:
        return domain
    if skill:
        return skill
    if section:
        return section

    question_text = (q.get("question_text") or "").strip()
    return question_text[:50] or "Unknown"


def _extract_algorithmic_features(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []

    for q in questions:
        prompt = q.get("prompt") or ""
        question_text = q.get("question_text") or ""
        answer_choices = q.get("answer_choices") or []
        explanation = q.get("explanation") or ""
        metadata = q.get("metadata") or {}

        answer_lengths = [len(str(choice)) for choice in answer_choices]

        skeleton_key, skeleton_name, skeleton_template = dataset_analysis.infer_skeleton_template(
            {"prompt": prompt, "question_text": question_text}
        )

        feature = {
            "id": q.get("id"),
            "topic": _infer_topic(q),
            "question_skeleton": skeleton_template,
            "question_skeleton_key": skeleton_key,
            "question_skeleton_name": skeleton_name,
            "answer_type": _infer_answer_type(q),
            "metadata": {
                "assessment": metadata.get("assessment"),
                "section": metadata.get("section"),
                "domain": metadata.get("domain"),
                "skill": metadata.get("skill"),
                "difficulty": metadata.get("difficulty"),
            },
            "lengths": {
                "prompt_chars": len(prompt),
                "prompt_words": len(prompt.split()),
                "question_text_chars": len(question_text),
                "question_text_words": len(question_text.split()),
                "question_total_chars": len(prompt) + len(question_text),
                "answer_choice_count": len(answer_choices),
                "answer_avg_chars": round(sum(answer_lengths) / len(answer_lengths), 2) if answer_lengths else 0,
                "answer_min_chars": min(answer_lengths) if answer_lengths else 0,
                "answer_max_chars": max(answer_lengths) if answer_lengths else 0,
                "explanation_chars": len(explanation),
                "explanation_words": len(explanation.split()),
            },
            "snippets": {
                "prompt_preview": _clean_text_snippet(prompt, 180),
                "question_text_preview": _clean_text_snippet(question_text, 120),
            },
        }
        features.append(feature)

    return features


def _extract_nodes(
    questions: List[Dict[str, Any]],
    batch_size: int = LLM_EXTRACTION_BATCH_SIZE,
    max_questions: int = LLM_EXTRACTION_MAX_QUESTIONS,
) -> List[Dict[str, Any]]:
    subset = questions[:max_questions]
    all_nodes: List[Dict[str, Any]] = []
    seen_patterns: set = set()

    if DEBUG_LOGS:
        print(
            f"[DEBUG][pipeline] llm_extraction_subset={len(subset)} "
            f"batch_size={batch_size} max_questions={max_questions}"
        )

    for i in range(0, len(subset), batch_size):
        batch = subset[i : i + batch_size]
        minified = []
        skill_counts: Dict[str, int] = defaultdict(int)
        for q in batch:
            skill = (q.get("metadata") or {}).get("skill") or "Unknown"
            skill_counts[str(skill)] += 1
            minified.append(
                {
                    "id": q.get("id"),
                    "prompt": q.get("prompt"),
                    "question_text": q.get("question_text"),
                    "correct_answer": q.get("correct_answer"),
                }
            )

        skill_hint = max(skill_counts.keys(), key=lambda k: skill_counts[k]) if skill_counts else "Mixed"
        if DEBUG_LOGS:
            print(
                f"[DEBUG][pipeline] llm_extract_batch index={i // batch_size + 1} "
                f"size={len(batch)} skill_hint={skill_hint}"
            )

        try:
            nodes = llm.extract_graph_nodes(minified, skill_hint)
        except Exception as exc:
            if DEBUG_LOGS:
                print(f"[DEBUG][pipeline] llm_extract_batch_error index={i // batch_size + 1} err={exc}")
            nodes = []

        for node in nodes:
            node_id = node.get("id") or (minified[0].get("id") if minified else None)
            node["id"] = node_id

            pattern_key = (
                node.get("topic", ""),
                node.get("logic_skeleton", ""),
                node.get("answer_skeleton", ""),
            )
            if pattern_key not in seen_patterns:
                all_nodes.append(node)
                seen_patterns.add(pattern_key)

    return all_nodes


def _compute_file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _processed_cache_dir(upload_path: str, use_llm: bool) -> str:
    file_hash = _compute_file_sha256(upload_path)
    key = f"{CACHE_VERSION}_hash_{file_hash}_llm_{int(use_llm)}_max_{LLM_EXTRACTION_MAX_QUESTIONS}_batch_{LLM_EXTRACTION_BATCH_SIZE}"
    return os.path.join(data_dir(), "processed", key)


def _cache_exists(cache_dir: str) -> bool:
    return all(os.path.isfile(os.path.join(cache_dir, artifact)) for artifact in CACHE_ARTIFACTS)


def _copy_artifacts(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for artifact in CACHE_ARTIFACTS:
        shutil.copyfile(os.path.join(src_dir, artifact), os.path.join(dst_dir, artifact))


def _features_to_seed_nodes(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nodes = []
    for f in features:
        nodes.append(
            {
                "id": f.get("id"),
                "topic": f.get("topic"),
                "logic_skeleton": f.get("question_skeleton"),
                "answer_skeleton": f.get("answer_type"),
            }
        )
    return nodes


def _build_catalogs(nodes: List[Dict[str, Any]], features: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[str]]:
    topics = set()
    logic_skeletons = set()
    answer_skeletons = set()
    question_skeletons = set()
    answer_types = set()

    for node in nodes:
        topic = node.get("Topic") or node.get("topic") or ""
        logic = node.get("Logic Skeleton") or node.get("logic_skeleton") or ""
        answer = node.get("Answer Skeleton") or node.get("answer_skeleton") or ""
        if topic:
            topics.add(topic)
        if logic:
            logic_skeletons.add(logic)
            question_skeletons.add(logic)
        if answer:
            answer_skeletons.add(answer)

    for feature in features or []:
        topic = feature.get("topic") or ""
        skeleton = feature.get("question_skeleton") or ""
        answer_type = feature.get("answer_type") or ""
        if topic:
            topics.add(topic)
        if skeleton:
            question_skeletons.add(skeleton)
            logic_skeletons.add(skeleton)
        if answer_type:
            answer_types.add(answer_type)
            answer_skeletons.add(answer_type)

    return {
        "topics": sorted(topics),
        "question_skeletons": sorted(question_skeletons),
        "answer_types": sorted(answer_types),
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


def _load_analysis_context(
    normalized: List[Dict[str, Any]],
    bundled_analysis: Optional[Dict[str, Any]] = None,
    analysis_file: Optional[str] = None,
) -> Dict[str, Any]:
    analysis: Dict[str, Any] = bundled_analysis or {}
    if analysis_file:
        if not os.path.isfile(analysis_file):
            raise PipelineError(f"analysis_file not found: {analysis_file}")
        with open(analysis_file, "r", encoding="utf-8") as f:
            analysis = json.load(f)

    if not analysis:
        analysis = dataset_analysis.analyze_questions(normalized)
    return analysis


def _generate_questions_from_features(
    normalized: List[Dict[str, Any]],
    features: List[Dict[str, Any]],
    question_topic: str,
    question_skeleton: str,
    answer_type: str,
    count: int,
    analysis: Dict[str, Any],
    user_context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    sample_question = _select_sample_question(normalized, features, question_topic, answer_type)
    style_profile = _style_profile_for_topic(features, question_topic, answer_type)
    style_profile["requested_metadata"] = _infer_requested_metadata(features, question_topic)
    dataset_context = _compact_generation_context(analysis)

    generated: List[Dict[str, Any]] = []
    seen_signatures: set = set()
    max_attempts_per_question = 5

    for index in range(count):
        generated_item: Optional[Dict[str, Any]] = None
        last_error: Optional[str] = None
        for attempt in range(1, max_attempts_per_question + 1):
            try:
                context = (user_context or "") + f"\nvariation_request: item={index + 1}, attempt={attempt}"
                raw = llm.generate_question_from_features(
                    question_topic=question_topic,
                    question_skeleton=question_skeleton,
                    answer_type=answer_type,
                    sample_question=sample_question,
                    style_profile=style_profile,
                    dataset_context=dataset_context,
                    user_context=context,
                )
            except Exception as exc:
                last_error = str(exc)
                if DEBUG_LOGS:
                    print(f"[DEBUG][pipeline] generation_error item={index + 1} attempt={attempt} err={exc}")
                continue

            candidate = _finalize_generated(raw, answer_type)
            if _is_too_similar(candidate, sample_question, seen_signatures):
                if DEBUG_LOGS:
                    print(f"[DEBUG][pipeline] rejected_similar item={index + 1} attempt={attempt}")
                continue

            generated_item = candidate
            signature = _signature_for_generated(generated_item)
            seen_signatures.add(signature)
            if DEBUG_LOGS:
                preview = (generated_item.get("question_text") or "")[:140]
                print(f"[DEBUG][pipeline] accepted item={index + 1} attempt={attempt} q_preview={preview}")
            break

        if generated_item is None:
            raise PipelineError(
                f"Failed to generate unique question {index + 1}/{count} after {max_attempts_per_question} attempts. "
                f"Last model error: {last_error or 'unknown'}"
            )

        generated_item["requested_topic"] = question_topic
        generated_item["requested_skeleton"] = question_skeleton
        generated_item["metadata"] = style_profile.get("requested_metadata", {})
        generated.append(generated_item)

    return generated


def _load_features_bundle(features_file: str) -> Dict[str, Any]:
    if not os.path.isfile(features_file):
        raise PipelineError(f"features_file not found: {features_file}")
    with open(features_file, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    if not isinstance(bundle.get("normalized_questions"), list) or not isinstance(bundle.get("question_features"), list):
        raise PipelineError("features_file must contain 'normalized_questions' and 'question_features' arrays.")
    return bundle


def _build_weighted_generation_plan(features: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    if not features:
        raise PipelineError("No question features found in features_file.")
    sampled = random.choices(features, k=count)
    plan: List[Dict[str, Any]] = []
    for item in sampled:
        plan.append(
            {
                "question_topic": item.get("topic") or "General",
                "question_skeleton": item.get("question_skeleton") or "General textual reasoning pattern.",
                "answer_type": item.get("answer_type") or "multiple_choice",
                "target_metadata": item.get("metadata") or {},
            }
        )
    return plan


def _finalize_problem_set_items(
    raw_items: List[Dict[str, Any]],
    plan: List[Dict[str, Any]],
    sample_question: Dict[str, Any],
) -> List[Dict[str, Any]]:
    finalized: List[Dict[str, Any]] = []
    for index, request in enumerate(plan):
        answer_type = str(request.get("answer_type") or "multiple_choice")
        raw = raw_items[index] if index < len(raw_items) and isinstance(raw_items[index], dict) else {}
        if not raw:
            raw = _fallback_generated_mc(sample_question)

        item = _finalize_generated(raw, answer_type)
        item["requested_topic"] = request.get("question_topic")
        item["requested_skeleton"] = request.get("question_skeleton")
        item["metadata"] = request.get("target_metadata") or {}
        finalized.append(item)
    return finalized


def _write_features_bundle(
    run_dir: str,
    normalized: List[Dict[str, Any]],
    insights: Dict[str, Any],
    features: List[Dict[str, Any]],
    graph_nodes: List[Dict[str, Any]],
    catalogs: Dict[str, List[str]],
    analysis: Dict[str, Any],
) -> None:
    bundle = {
        "normalized_questions": normalized,
        "insights": insights,
        "question_features": features,
        "graph_nodes": graph_nodes,
        "catalogs": catalogs,
        "dataset_analysis": analysis,
    }
    save_json(os.path.join(run_dir, "features.json"), bundle)


def _compact_generation_context(analysis: Dict[str, Any]) -> Dict[str, Any]:
    skeletons = (analysis.get("question_skeleton_catalog", {}) or {}).get("observed", [])[:12]
    return {
        "dataset_size": analysis.get("dataset_size", 0),
        "metadata_summary": analysis.get("metadata_summary", {}),
        "length_summary": analysis.get("length_summary", {}),
        "core_topics": (analysis.get("core_topics", {}) or {}).get("groups", [])[:12],
        "question_skeletons": [
            {
                "key": item.get("key"),
                "name": item.get("name"),
                "template": item.get("template"),
                "count": item.get("count"),
            }
            for item in skeletons
        ],
    }


def _infer_requested_metadata(features: List[Dict[str, Any]], question_topic: str) -> Dict[str, str]:
    matched: List[Dict[str, Any]] = []
    for feature in features:
        if (feature.get("topic") or "") == question_topic:
            matched.append(feature)

    if not matched:
        matched = features

    def most_common(field: str, fallback: str = "Unknown") -> str:
        counts: Dict[str, int] = defaultdict(int)
        for item in matched:
            meta = item.get("metadata") or {}
            value = str(meta.get(field) or "").strip()
            if value:
                counts[value] += 1
        if not counts:
            return fallback
        return max(counts, key=lambda key: counts[key])

    return {
        "assessment": "SAT",
        "section": most_common("section", "Reading and Writing"),
        "domain": most_common("domain"),
        "skill": most_common("skill"),
        "difficulty": most_common("difficulty", "Medium"),
    }


def _fallback_generated_mc(sample_question: Dict[str, Any]) -> Dict[str, Any]:
    answer_choices = sample_question.get("answer_choices") or []
    fallback_choices = answer_choices[:4] if len(answer_choices) >= 4 else ["Choice A", "Choice B", "Choice C", "Choice D"]
    return {
        "prompt": sample_question.get("prompt") or "Read the passage and answer the question.",
        "question_text": sample_question.get("question_text") or "Which choice best answers the question?",
        "correct_answer_text": fallback_choices[0],
        "distractors": fallback_choices[1:4],
        "explanation": "Fallback output used because model generation failed.",
    }


def _normalize_text_for_compare(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").lower()).strip()
    cleaned = re.sub(r"[^a-z0-9 ]", "", cleaned)
    return cleaned


def _word_overlap_ratio(a: str, b: str) -> float:
    set_a = {w for w in _normalize_text_for_compare(a).split(" ") if w}
    set_b = {w for w in _normalize_text_for_compare(b).split(" ") if w}
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union else 0.0


def _is_too_similar(generated: Dict[str, Any], sample_question: Dict[str, Any], prior_signatures: set) -> bool:
    generated_prompt = str(generated.get("prompt") or "")
    generated_question = str(generated.get("question_text") or "")
    combined = f"{generated_prompt}\n{generated_question}".strip()
    signature = _normalize_text_for_compare(combined)

    if signature in prior_signatures:
        return True

    sample_prompt = str(sample_question.get("prompt") or "")
    sample_question_text = str(sample_question.get("question_text") or "")
    sample_combined = f"{sample_prompt}\n{sample_question_text}".strip()

    overlap = _word_overlap_ratio(combined, sample_combined)
    if overlap >= 0.70:
        return True

    return False


def _signature_for_generated(generated: Dict[str, Any]) -> str:
    combined = f"{generated.get('prompt') or ''}\n{generated.get('question_text') or ''}"
    return _normalize_text_for_compare(combined)


def _finalize_multiple_choice(raw: Dict[str, Any]) -> Dict[str, Any]:
    correct_text = str(raw.get("correct_answer_text") or "")
    distractors = raw.get("distractors") or []

    if not isinstance(distractors, list):
        distractors = []

    distractors = [str(x) for x in distractors if str(x).strip()]
    if len(distractors) < 3:
        while len(distractors) < 3:
            distractors.append(f"Distractor {len(distractors) + 1}")
    distractors = distractors[:3]

    all_choices = distractors + [correct_text or "Correct answer"]
    random.shuffle(all_choices)

    correct_index = all_choices.index(correct_text or "Correct answer")
    correct_letter = ["A", "B", "C", "D"][correct_index]

    return {
        "prompt": str(raw.get("prompt") or ""),
        "question_text": str(raw.get("question_text") or ""),
        "answer_type": "multiple_choice",
        "answer_choices": all_choices,
        "correct_answer": correct_letter,
        "explanation": str(raw.get("explanation") or ""),
    }


def _finalize_free_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    expected_answer = raw.get("expected_answer") or raw.get("correct_answer_text") or ""
    return {
        "prompt": str(raw.get("prompt") or ""),
        "question_text": str(raw.get("question_text") or ""),
        "answer_type": "free_response",
        "expected_answer": str(expected_answer),
        "explanation": str(raw.get("explanation") or ""),
    }


def _finalize_generated(raw: Dict[str, Any], answer_type: str) -> Dict[str, Any]:
    if answer_type == "free_response":
        return _finalize_free_response(raw)
    return _finalize_multiple_choice(raw)


def _style_profile_for_topic(features: List[Dict[str, Any]], topic: str, answer_type: str) -> Dict[str, Any]:
    matched = [f for f in features if f.get("topic") == topic and f.get("answer_type") == answer_type]
    if not matched:
        matched = [f for f in features if f.get("answer_type") == answer_type]
    if not matched:
        matched = features

    prompt_lengths = [f.get("lengths", {}).get("prompt_chars", 0) for f in matched]
    question_lengths = [f.get("lengths", {}).get("question_text_chars", 0) for f in matched]
    answer_lengths = [f.get("lengths", {}).get("answer_avg_chars", 0) for f in matched]

    def _avg(values: List[float]) -> float:
        if not values:
            return 0
        return round(sum(values) / len(values), 2)

    return {
        "topic": topic,
        "answer_type": answer_type,
        "sample_count": len(matched),
        "avg_prompt_chars": _avg(prompt_lengths),
        "avg_question_text_chars": _avg(question_lengths),
        "avg_answer_chars": _avg(answer_lengths),
    }


def _select_sample_question(
    normalized_questions: List[Dict[str, Any]],
    features: List[Dict[str, Any]],
    topic: str,
    answer_type: str,
) -> Dict[str, Any]:
    feature_by_id = {f.get("id"): f for f in features}

    candidates = []
    for q in normalized_questions:
        f = feature_by_id.get(q.get("id"))
        if not f:
            continue
        if f.get("topic") == topic and f.get("answer_type") == answer_type:
            candidates.append(q)

    if not candidates:
        for q in normalized_questions:
            f = feature_by_id.get(q.get("id"))
            if not f:
                continue
            if f.get("topic") == topic:
                candidates.append(q)

    if not candidates:
        candidates = normalized_questions

    if not candidates:
        raise PipelineError("No questions available to build a sample prompt.")

    return {
        "prompt": candidates[0].get("prompt"),
        "question_text": candidates[0].get("question_text"),
        "answer_choices": candidates[0].get("answer_choices"),
        "correct_answer": candidates[0].get("correct_answer"),
        "explanation": candidates[0].get("explanation"),
    }


def run_feature_extraction(upload_path: str, run_id: str, use_llm: bool = True) -> Dict[str, Any]:
    run_dir = create_run_dir(run_id)

    cache_dir = _processed_cache_dir(upload_path, use_llm)
    if _cache_exists(cache_dir):
        if DEBUG_LOGS:
            print(f"[DEBUG][pipeline] cache_hit dir={cache_dir}")
        _copy_artifacts(cache_dir, run_dir)

        with open(os.path.join(run_dir, "input_normalized.json"), "r", encoding="utf-8") as f:
            normalized = json.load(f)

        summary = {
            "run_id": run_id,
            "input_count": len(normalized),
            "phase": "feature_extraction",
            "cache_hit": True,
            "cache_key": os.path.basename(cache_dir),
            "artifacts": {
                "input_normalized": "input_normalized.json",
                "insights": "insights.json",
                "question_features": "question_features.json",
                "graph_nodes": "graph_nodes.json",
                "catalogs": "catalogs.json",
                "dataset_analysis": "dataset_analysis.json",
                "features": "features.json",
            },
        }
        save_json(os.path.join(run_dir, "run.json"), summary)
        return summary

    questions_raw = _load_questions(upload_path)
    normalized = [_normalize_question(q, i) for i, q in enumerate(questions_raw)]
    _validate_questions(normalized)

    save_json(os.path.join(run_dir, "input_normalized.json"), normalized)

    insights = _compute_insights(normalized)
    save_json(os.path.join(run_dir, "insights.json"), insights)

    features = _extract_algorithmic_features(normalized)
    save_json(os.path.join(run_dir, "question_features.json"), features)

    analysis = dataset_analysis.analyze_questions(normalized)
    save_json(os.path.join(run_dir, "dataset_analysis.json"), analysis)

    seed_nodes = _features_to_seed_nodes(features)
    llm_nodes = _extract_nodes(normalized) if use_llm else []
    graph_nodes = seed_nodes + llm_nodes
    save_json(os.path.join(run_dir, "graph_nodes.json"), graph_nodes)

    catalogs = _build_catalogs(graph_nodes, features)
    save_json(os.path.join(run_dir, "catalogs.json"), catalogs)
    _write_features_bundle(
        run_dir=run_dir,
        normalized=normalized,
        insights=insights,
        features=features,
        graph_nodes=graph_nodes,
        catalogs=catalogs,
        analysis=analysis,
    )

    summary = {
        "run_id": run_id,
        "input_count": len(normalized),
        "phase": "feature_extraction",
        "cache_hit": False,
        "cache_key": os.path.basename(cache_dir),
        "artifacts": {
            "input_normalized": "input_normalized.json",
            "insights": "insights.json",
            "question_features": "question_features.json",
            "graph_nodes": "graph_nodes.json",
            "catalogs": "catalogs.json",
            "dataset_analysis": "dataset_analysis.json",
            "features": "features.json",
        },
    }

    os.makedirs(cache_dir, exist_ok=True)
    _copy_artifacts(run_dir, cache_dir)
    if DEBUG_LOGS:
        print(f"[DEBUG][pipeline] cache_written dir={cache_dir}")

    save_json(os.path.join(run_dir, "run.json"), summary)
    return summary


def run_question_generation(
    feature_run_id: str,
    run_id: str,
    question_topic: str,
    question_skeleton: str,
    answer_type: str,
    count: int,
    analysis_file: Optional[str] = None,
    user_context: Optional[str] = None,
) -> Dict[str, Any]:
    if count < 1:
        raise PipelineError("count must be >= 1")

    source_dir = run_path(feature_run_id)
    if not os.path.isdir(source_dir):
        raise PipelineError(f"Feature run not found: {feature_run_id}")

    input_path = os.path.join(source_dir, "input_normalized.json")
    features_path = os.path.join(source_dir, "question_features.json")
    if not os.path.isfile(input_path) or not os.path.isfile(features_path):
        raise PipelineError("Feature run is missing required artifacts (input_normalized.json/question_features.json).")

    with open(input_path, "r", encoding="utf-8") as f:
        normalized = json.load(f)
    with open(features_path, "r", encoding="utf-8") as f:
        features = json.load(f)

    bundled_analysis: Dict[str, Any] = {}
    default_analysis_path = os.path.join(source_dir, "dataset_analysis.json")
    if os.path.isfile(default_analysis_path):
        with open(default_analysis_path, "r", encoding="utf-8") as f:
            bundled_analysis = json.load(f)

    analysis = _load_analysis_context(
        normalized=normalized,
        bundled_analysis=bundled_analysis,
        analysis_file=analysis_file,
    )
    if not os.path.isfile(default_analysis_path) and not analysis_file:
        save_json(default_analysis_path, analysis)

    generated = _generate_questions_from_features(
        normalized=normalized,
        features=features,
        question_topic=question_topic,
        question_skeleton=question_skeleton,
        answer_type=answer_type,
        count=count,
        analysis=analysis,
        user_context=user_context,
    )

    run_dir = create_run_dir(run_id)
    save_json(os.path.join(run_dir, "generation_request.json"), {
        "feature_run_id": feature_run_id,
        "question_topic": question_topic,
        "question_skeleton": question_skeleton,
        "answer_type": answer_type,
        "count": count,
        "analysis_file": analysis_file,
    })
    save_json(os.path.join(run_dir, "generated_questions.json"), generated)

    summary = {
        "run_id": run_id,
        "phase": "question_generation",
        "source_feature_run_id": feature_run_id,
        "generated_count": len(generated),
        "artifacts": {
            "generation_request": "generation_request.json",
            "generated_questions": "generated_questions.json",
        },
    }
    save_json(os.path.join(run_dir, "run.json"), summary)
    return summary


def run_question_generation_from_features_file(
    features_file: str,
    run_id: str,
    question_topic: str,
    question_skeleton: str,
    answer_type: str,
    count: int,
    analysis_file: Optional[str] = None,
    user_context: Optional[str] = None,
) -> Dict[str, Any]:
    if count < 1:
        raise PipelineError("count must be >= 1")

    if not os.path.isfile(features_file):
        raise PipelineError(f"features_file not found: {features_file}")

    with open(features_file, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    normalized = bundle.get("normalized_questions")
    features = bundle.get("question_features")
    bundled_analysis = bundle.get("dataset_analysis") or {}

    if not isinstance(normalized, list) or not isinstance(features, list):
        raise PipelineError("features_file must contain 'normalized_questions' and 'question_features' arrays.")

    analysis = _load_analysis_context(
        normalized=normalized,
        bundled_analysis=bundled_analysis,
        analysis_file=analysis_file,
    )

    generated = _generate_questions_from_features(
        normalized=normalized,
        features=features,
        question_topic=question_topic,
        question_skeleton=question_skeleton,
        answer_type=answer_type,
        count=count,
        analysis=analysis,
        user_context=user_context,
    )

    run_dir = create_run_dir(run_id)
    save_json(
        os.path.join(run_dir, "generation_request.json"),
        {
            "features_file": features_file,
            "question_topic": question_topic,
            "question_skeleton": question_skeleton,
            "answer_type": answer_type,
            "count": count,
            "analysis_file": analysis_file,
        },
    )
    save_json(os.path.join(run_dir, "generated_questions.json"), generated)

    summary = {
        "run_id": run_id,
        "phase": "question_generation",
        "source_features_file": features_file,
        "generated_count": len(generated),
        "artifacts": {
            "generation_request": "generation_request.json",
            "generated_questions": "generated_questions.json",
        },
    }
    save_json(os.path.join(run_dir, "run.json"), summary)
    return summary


def run_random_question_from_features_file(
    features_file: str,
    run_id: str,
    analysis_file: Optional[str] = None,
    user_context: Optional[str] = None,
) -> Dict[str, Any]:
    bundle = _load_features_bundle(features_file)
    normalized_raw = bundle.get("normalized_questions")
    features_raw = bundle.get("question_features")
    bundled_analysis = bundle.get("dataset_analysis") or {}

    if not isinstance(normalized_raw, list) or not isinstance(features_raw, list):
        raise PipelineError("features_file must contain 'normalized_questions' and 'question_features' arrays.")

    normalized: List[Dict[str, Any]] = normalized_raw
    features: List[Dict[str, Any]] = features_raw

    plan = _build_weighted_generation_plan(features, 1)
    request = plan[0]
    analysis = _load_analysis_context(
        normalized=normalized,
        bundled_analysis=bundled_analysis,
        analysis_file=analysis_file,
    )

    generated = _generate_questions_from_features(
        normalized=normalized,
        features=features,
        question_topic=str(request.get("question_topic") or "General"),
        question_skeleton=str(request.get("question_skeleton") or "General textual reasoning pattern."),
        answer_type=str(request.get("answer_type") or "multiple_choice"),
        count=1,
        analysis=analysis,
        user_context=user_context,
    )

    run_dir = create_run_dir(run_id)
    save_json(
        os.path.join(run_dir, "generation_request.json"),
        {
            "features_file": features_file,
            "mode": "random_question",
            "analysis_file": analysis_file,
        },
    )
    save_json(os.path.join(run_dir, "generated_questions.json"), generated)

    summary = {
        "run_id": run_id,
        "phase": "question_generation",
        "mode": "random_question",
        "source_features_file": features_file,
        "generated_count": 1,
        "generated_questions": generated,
        "artifacts": {
            "generation_request": "generation_request.json",
            "generated_questions": "generated_questions.json",
        },
    }
    save_json(os.path.join(run_dir, "run.json"), summary)
    return summary


def run_problem_set_generation_from_features_file(
    features_file: str,
    run_id: str,
    count: int,
    analysis_file: Optional[str] = None,
    user_context: Optional[str] = None,
) -> Dict[str, Any]:
    if count < 1:
        raise PipelineError("count must be >= 1")

    bundle = _load_features_bundle(features_file)
    normalized_raw = bundle.get("normalized_questions")
    features_raw = bundle.get("question_features")
    bundled_analysis = bundle.get("dataset_analysis") or {}

    if not isinstance(normalized_raw, list) or not isinstance(features_raw, list):
        raise PipelineError("features_file must contain 'normalized_questions' and 'question_features' arrays.")

    normalized: List[Dict[str, Any]] = normalized_raw
    features: List[Dict[str, Any]] = features_raw

    analysis = _load_analysis_context(
        normalized=normalized,
        bundled_analysis=bundled_analysis,
        analysis_file=analysis_file,
    )
    plan = _build_weighted_generation_plan(features, count)
    sample_question = _pick_example(normalized)
    dataset_context = _compact_generation_context(analysis)

    raw_items: List[Dict[str, Any]] = []
    try:
        raw_items = llm.generate_question_set_from_plan(
            plan=plan,
            sample_question=sample_question,
            dataset_context=dataset_context,
            user_context=user_context,
        )
    except Exception as exc:
        if DEBUG_LOGS:
            print(f"[DEBUG][pipeline] problem_set_batch_generation_error err={exc}")

    if not raw_items:
        raw_items = []

    generated = _finalize_problem_set_items(raw_items=raw_items, plan=plan, sample_question=sample_question)

    run_dir = create_run_dir(run_id)
    save_json(
        os.path.join(run_dir, "generation_request.json"),
        {
            "features_file": features_file,
            "mode": "problem_set",
            "count": count,
            "analysis_file": analysis_file,
        },
    )
    save_json(os.path.join(run_dir, "generated_questions.json"), generated)

    summary = {
        "run_id": run_id,
        "phase": "question_generation",
        "mode": "problem_set",
        "source_features_file": features_file,
        "generated_count": len(generated),
        "generated_questions": generated,
        "artifacts": {
            "generation_request": "generation_request.json",
            "generated_questions": "generated_questions.json",
        },
    }
    save_json(os.path.join(run_dir, "run.json"), summary)
    return summary


def _generate_questions(
    catalogs: Dict[str, List[str]],
    example_question: Dict[str, Any],
    count: int,
    dataset_context: Optional[Dict[str, Any]] = None,
    user_context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    topics = catalogs.get("topics") or []
    logic_skeletons = catalogs.get("logic_skeletons") or catalogs.get("question_skeletons") or []
    answer_skeletons = catalogs.get("answer_skeletons") or catalogs.get("answer_types") or ["multiple_choice"]

    if not topics or not logic_skeletons:
        raise PipelineError("Insufficient extracted data to generate questions.")

    generated = []
    seen_signatures: set = set()
    max_attempts_per_question = 5

    for index in range(count):
        generated_item: Optional[Dict[str, Any]] = None
        last_error: Optional[str] = None

        for attempt in range(1, max_attempts_per_question + 1):
            topic = random.choice(topics)
            logic = random.choice(logic_skeletons)
            answer = random.choice(answer_skeletons)

            if DEBUG_LOGS:
                print(
                    f"[DEBUG][pipeline] generate_loop item={index + 1} attempt={attempt} "
                    f"topic={topic} logic={logic} answer={answer}"
                )

            try:
                context = (user_context or "") + f"\nvariation_request: item={index + 1}, attempt={attempt}"
                raw = llm.generate_question(
                    topic,
                    logic,
                    answer,
                    example_question,
                    dataset_context=dataset_context,
                    user_context=context,
                )
            except Exception as exc:
                last_error = str(exc)
                if DEBUG_LOGS:
                    print(f"[DEBUG][pipeline] generation_error item={index + 1} attempt={attempt} err={exc}")
                continue

            candidate = _finalize_multiple_choice(raw)
            if _is_too_similar(candidate, example_question, seen_signatures):
                if DEBUG_LOGS:
                    print(f"[DEBUG][pipeline] rejected_similar item={index + 1} attempt={attempt}")
                continue

            generated_item = candidate
            seen_signatures.add(_signature_for_generated(candidate))
            break

        if generated_item is None:
            raise PipelineError(
                f"Failed to generate unique question {index + 1}/{count} after {max_attempts_per_question} attempts. "
                f"Last model error: {last_error or 'unknown'}"
            )

        generated.append(generated_item)

    return generated


def run_pipeline(
    upload_path: str,
    run_id: str,
    extract_nodes: bool,
    generate_count: int,
    user_id: Optional[str] = None,
    memory_query: Optional[str] = None,
) -> Dict[str, Any]:
    summary = run_feature_extraction(upload_path=upload_path, run_id=run_id, use_llm=extract_nodes or generate_count > 0)

    run_dir = run_path(run_id)
    with open(os.path.join(run_dir, "input_normalized.json"), "r", encoding="utf-8") as f:
        normalized = json.load(f)
    with open(os.path.join(run_dir, "catalogs.json"), "r", encoding="utf-8") as f:
        catalogs = json.load(f)
    analysis_context: Dict[str, Any] = {}
    analysis_path = os.path.join(run_dir, "dataset_analysis.json")
    if os.path.isfile(analysis_path):
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis_context = _compact_generation_context(json.load(f))

    generated: List[Dict[str, Any]] = []
    if generate_count > 0:
        user_context = memory_query if memory_query else ""
        example = _pick_example(normalized)
        generated = _generate_questions(
            catalogs,
            example,
            generate_count,
            dataset_context=analysis_context,
            user_context=user_context,
        )
        save_json(os.path.join(run_dir, "generated_questions.json"), generated)

    compat_summary = {
        "run_id": run_id,
        "input_count": len(normalized),
        "extract_nodes": extract_nodes,
        "generate_count": generate_count,
        "user_id": user_id,
        "artifacts": {
            "input_normalized": "input_normalized.json",
            "insights": "insights.json",
            "question_features": "question_features.json",
            "graph_nodes": "graph_nodes.json" if extract_nodes or generate_count > 0 else None,
            "catalogs": "catalogs.json" if extract_nodes or generate_count > 0 else None,
            "dataset_analysis": "dataset_analysis.json",
            "generated_questions": "generated_questions.json" if generate_count > 0 else None,
        },
    }

    save_json(os.path.join(run_dir, "run.json"), compat_summary)
    return compat_summary
