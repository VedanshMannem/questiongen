import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _word_count(text: str) -> int:
    return len([w for w in _compact(text).split(" ") if w])


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "average": 0}
    return {
        "count": len(values),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "average": round(sum(values) / len(values), 2),
    }


SKELETON_LIBRARY: List[Dict[str, Any]] = [
    {
        "key": "logical_completion",
        "name": "Cause-and-Consequence Logical Completion",
        "template": "A passage presents context and constraints, then leaves a blank. Select the choice that most logically completes the idea while preserving causal consistency.",
        "cues": ["most logically completes", "____", " blank"],
    },
    {
        "key": "main_idea",
        "name": "Main Idea Synthesis",
        "template": "A passage introduces several details and examples. Determine the claim that best synthesizes those details into a single central idea.",
        "cues": ["main idea", "most accurately states"],
    },
    {
        "key": "supports_claim",
        "name": "Evidence That Supports a Claim",
        "template": "A claim or hypothesis is provided. Identify the finding that, if true, would most directly strengthen that claim.",
        "cues": ["would support", "most directly support", "if true, would support"],
    },
    {
        "key": "weakens_claim",
        "name": "Evidence That Weakens a Claim",
        "template": "A researcher claim is presented. Choose the result that most directly weakens or challenges that claim.",
        "cues": ["would weaken", "most directly weaken", "if true, would weaken"],
    },
    {
        "key": "data_completion",
        "name": "Data-Driven Statement Completion",
        "template": "A table or graph is provided with a partial statement. Use quantitative evidence to complete the statement with the most accurate comparison or value.",
        "cues": ["uses data", "from the table", "from the graph", "data from"],
    },
    {
        "key": "best_inference",
        "name": "Best-Supported Inference",
        "template": "A text presents facts and constraints. Infer the conclusion that is best supported without introducing unsupported assumptions.",
        "cues": ["can be inferred", "best supported", "logically follows"],
    },
    {
        "key": "methodology_interpretation",
        "name": "Research Design and Interpretation",
        "template": "A study setup and findings are described. Determine what methodological conclusion is justified by how variables and outcomes relate.",
        "cues": ["researchers", "study", "experiment", "conclusion"],
    },
    {
        "key": "author_perspective",
        "name": "Author or Historian Perspective",
        "template": "A viewpoint is contrasted with alternatives. Identify the statement that best captures the author’s or scholars’ stance.",
        "cues": ["according to", "historians", "journalist", "scholars"],
    },
    {
        "key": "quotation_selection",
        "name": "Quotation That Best Illustrates a Claim",
        "template": "A claim about language or interpretation is given. Select the quotation that most directly exemplifies that claim.",
        "cues": ["quotation", "best support", "illustrates the claim"],
    },
    {
        "key": "comparative_reasoning",
        "name": "Cross-Case Comparative Reasoning",
        "template": "Two entities, periods, or datasets are compared. Determine which statement correctly characterizes their relationship.",
        "cues": ["compared to", "difference between", "whereas", "than"],
    },
    {
        "key": "vocabulary_in_context",
        "name": "Vocabulary-in-Context Precision",
        "template": "A word or phrase appears in context. Select the meaning or replacement that best preserves the intended nuance.",
        "cues": ["as used in", "most nearly means", "best replacement"],
    },
    {
        "key": "general_reasoning",
        "name": "General Textual Reasoning",
        "template": "A passage includes claims and supporting details. Choose the answer that most logically aligns with the passage’s reasoning.",
        "cues": [],
    },
]


TOPIC_TAXONOMY: Dict[str, List[str]] = {
    "Animals and Ecology": [
        "species",
        "animal",
        "birds",
        "fish",
        "lizard",
        "frog",
        "octopus",
        "ecolog",
        "forest",
        "plant",
        "deer",
    ],
    "Archaeology and Ancient History": [
        "fossil",
        "archae",
        "ancient",
        "cambrian",
        "pyramid",
        "prehistoric",
    ],
    "Literature and Arts": [
        "novel",
        "poem",
        "artist",
        "curator",
        "theater",
        "film",
        "narrative",
        "photography",
    ],
    "Social Science and Behavior": [
        "survey",
        "participants",
        "social",
        "behavior",
        "marketing",
        "residents",
        "gift",
    ],
    "Economics and Policy": [
        "economic",
        "policy",
        "trade",
        "uncertainty",
        "public spending",
        "tax",
    ],
    "STEM Research and Methods": [
        "research",
        "experiment",
        "data",
        "graph",
        "table",
        "temperature",
        "genetic",
        "nucleobase",
        "transposon",
    ],
}


def _classify_skeleton(question: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _compact(str(question.get("prompt") or "")).lower()
    question_text = _compact(str(question.get("question_text") or "")).lower()
    full_text = f"{prompt} {question_text}"

    for skeleton in SKELETON_LIBRARY:
        cues = skeleton["cues"]
        if cues and any(cue in full_text for cue in cues):
            return skeleton

    if any(token in full_text for token in ["graph", "table", "data"]):
        return next(x for x in SKELETON_LIBRARY if x["key"] == "data_completion")
    if "support" in full_text:
        return next(x for x in SKELETON_LIBRARY if x["key"] == "supports_claim")
    if "weaken" in full_text:
        return next(x for x in SKELETON_LIBRARY if x["key"] == "weakens_claim")
    return next(x for x in SKELETON_LIBRARY if x["key"] == "general_reasoning")


def infer_skeleton_template(question: Dict[str, Any]) -> Tuple[str, str, str]:
    skeleton = _classify_skeleton(question)
    return str(skeleton["key"]), str(skeleton["name"]), str(skeleton["template"])


def _extract_topic_labels(question: Dict[str, Any]) -> List[str]:
    prompt = _compact(str(question.get("prompt") or "")).lower()
    question_text = _compact(str(question.get("question_text") or "")).lower()
    full_text = f"{prompt} {question_text}"

    labels: List[str] = []
    for topic, keywords in TOPIC_TAXONOMY.items():
        if any(keyword in full_text for keyword in keywords):
            labels.append(topic)

    if not labels:
        domain = ((question.get("metadata") or {}).get("domain") or "Unknown Domain").strip()
        labels.append(f"Domain-Driven: {domain}")

    return labels


def analyze_questions(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt_lengths: List[int] = []
    question_text_lengths: List[int] = []
    total_lengths: List[int] = []
    explanation_lengths: List[int] = []
    answer_choice_counts: List[int] = []
    answer_choice_lengths: List[int] = []

    domain_counter: Counter[str] = Counter()
    skill_counter: Counter[str] = Counter()
    difficulty_counter: Counter[str] = Counter()
    section_counter: Counter[str] = Counter()
    answer_type_counter: Counter[str] = Counter()

    skeleton_counts: Counter[str] = Counter()
    skeleton_examples: Dict[str, List[str]] = defaultdict(list)
    topic_counter: Counter[str] = Counter()

    for q in questions:
        prompt = str(q.get("prompt") or "")
        question_text = str(q.get("question_text") or "")
        explanation = str(q.get("explanation") or "")
        answers = q.get("answer_choices") or []
        metadata = q.get("metadata") or {}

        prompt_lengths.append(len(prompt))
        question_text_lengths.append(len(question_text))
        total_lengths.append(len(prompt) + len(question_text))
        explanation_lengths.append(len(explanation))

        answer_choice_counts.append(len(answers) if isinstance(answers, list) else 0)
        if isinstance(answers, list):
            for a in answers:
                answer_choice_lengths.append(len(str(a)))

        domain_counter[str(metadata.get("domain") or "Unknown")] += 1
        skill_counter[str(metadata.get("skill") or "Unknown")] += 1
        difficulty_counter[str(metadata.get("difficulty") or "Unknown")] += 1
        section_counter[str(metadata.get("section") or "Unknown")] += 1

        answer_type = "multiple_choice" if isinstance(answers, list) and len(answers) >= 2 else "free_response"
        answer_type_counter[answer_type] += 1

        skeleton = _classify_skeleton(q)
        skeleton_key = str(skeleton["key"])
        skeleton_counts[skeleton_key] += 1
        if len(skeleton_examples[skeleton_key]) < 4:
            skeleton_examples[skeleton_key].append(_compact(question_text)[:140])

        for topic in _extract_topic_labels(q):
            topic_counter[topic] += 1

    skeletons_observed: List[Dict[str, Any]] = []
    skeleton_by_key = {item["key"]: item for item in SKELETON_LIBRARY}
    for key, count in skeleton_counts.most_common():
        template = skeleton_by_key.get(key, {})
        skeletons_observed.append(
            {
                "key": key,
                "name": template.get("name", key),
                "template": template.get("template", ""),
                "count": count,
                "examples": skeleton_examples.get(key, []),
            }
        )

    return {
        "dataset_size": len(questions),
        "metadata_summary": {
            "domains": dict(domain_counter.most_common()),
            "skills": dict(skill_counter.most_common()),
            "difficulties": dict(difficulty_counter.most_common()),
            "sections": dict(section_counter.most_common()),
            "answer_types": dict(answer_type_counter.most_common()),
        },
        "length_summary": {
            "prompt_chars": _stats([float(x) for x in prompt_lengths]),
            "question_text_chars": _stats([float(x) for x in question_text_lengths]),
            "question_total_chars": _stats([float(x) for x in total_lengths]),
            "explanation_chars": _stats([float(x) for x in explanation_lengths]),
            "answer_choice_count": _stats([float(x) for x in answer_choice_counts]),
            "answer_choice_chars": _stats([float(x) for x in answer_choice_lengths]),
            "prompt_words": _stats([float(_word_count(str(q.get("prompt") or ""))) for q in questions]),
            "question_words": _stats([float(_word_count(str(q.get("question_text") or ""))) for q in questions]),
        },
        "core_topics": {
            "groups": [
                {"topic": topic, "count": count, "ratio": round(count / max(1, len(questions)), 4)}
                for topic, count in topic_counter.most_common()
            ]
        },
        "question_skeleton_catalog": {
            "observed": skeletons_observed,
            "library": [
                {
                    "key": item["key"],
                    "name": item["name"],
                    "template": item["template"],
                }
                for item in SKELETON_LIBRARY
            ],
        },
    }


def analyze_json_file(input_path: str, output_path: str) -> Dict[str, Any]:
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "questions" in raw:
        questions = raw["questions"]
    else:
        questions = raw

    if not isinstance(questions, list):
        raise ValueError("Input must be a list of question objects or an object with a 'questions' list.")

    analysis = analyze_questions(questions)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    return analysis
