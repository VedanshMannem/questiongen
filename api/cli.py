import argparse
import json
import os
from typing import Optional

from . import dataset_analysis
from .pipeline import (
    _compute_insights,
    _load_questions,
    _normalize_question,
    _validate_questions,
    run_feature_extraction,
    run_question_generation_from_features_file,
)
from .storage import new_run_id, uploads_dir


def _find_latest_json(folder: str) -> Optional[str]:
    if not os.path.isdir(folder):
        return None

    candidates = []
    for name in os.listdir(folder):
        if name.lower().endswith(".json"):
            path = os.path.join(folder, name)
            candidates.append(path)

    if not candidates:
        return None

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _dry_run(input_path: str) -> None:
    questions_raw = _load_questions(input_path)
    normalized = [_normalize_question(q, i) for i, q in enumerate(questions_raw)]
    _validate_questions(normalized)
    insights = _compute_insights(normalized)
    analysis = dataset_analysis.analyze_questions(normalized)

    print("Dry run passed: dataset is valid.")
    print(f"- questions: {len(normalized)}")
    print(f"- domains: {len((analysis.get('metadata_summary', {}) or {}).get('domains', {}))}")
    print(f"- skills: {len((analysis.get('metadata_summary', {}) or {}).get('skills', {}))}")
    print(f"- skeleton families: {len((analysis.get('question_skeleton_catalog', {}) or {}).get('observed', []))}")

    overall = (insights.get("overall") or {}).get("question_length") or {}
    if overall:
        print(
            f"- question length (chars): avg={overall.get('average')} min={overall.get('min')} max={overall.get('max')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="SAT question pipeline CLI")
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract-nodes", help="Extract algorithmic + AI features bundle")
    extract_parser.add_argument("--input", help="Path to a JSON dataset file.")
    extract_parser.add_argument("--input-folder", default=uploads_dir(), help="Folder to scan when --input is omitted.")
    extract_parser.add_argument("--use-llm", action="store_true", help="Use LLM for node extraction.")
    extract_parser.add_argument("--dry-run", action="store_true", help="Validate dataset only.")

    create_parser = subparsers.add_parser("create-questions", help="Create questions from a features.json file")
    create_parser.add_argument("--features-file", required=True, help="Path to features.json")
    create_parser.add_argument("--question-topic", required=True)
    create_parser.add_argument("--question-skeleton", required=True)
    create_parser.add_argument("--answer-type", default="multiple_choice", choices=["multiple_choice", "free_response"])
    create_parser.add_argument("--count", type=int, default=1)
    create_parser.add_argument("--analysis-file", help="Optional analysis JSON path")
    create_parser.add_argument("--user-context", help="Optional user context")

    legacy_parser = subparsers.add_parser("run", help="Legacy single-run flow")
    legacy_parser.add_argument("--input", help="Path to a JSON dataset file.")
    legacy_parser.add_argument("--input-folder", default=uploads_dir(), help="Folder to scan when --input is omitted.")
    legacy_parser.add_argument("--extract-nodes", action="store_true", help="Extract graph nodes.")
    legacy_parser.add_argument("--generate-count", type=int, default=0, help="Number of questions to generate.")
    legacy_parser.add_argument("--dry-run", action="store_true", help="Validate and summarize dataset without writing artifacts.")

    args = parser.parse_args()

    if args.command in {"extract-nodes", "run"}:
        input_path = args.input
        if not input_path:
            input_path = _find_latest_json(args.input_folder)

        if not input_path:
            raise SystemExit("No input JSON file found.")

        if not os.path.isfile(input_path):
            raise SystemExit(f"Input file not found: {input_path}")

        if args.dry_run:
            _dry_run(input_path)
            return

        run_id = new_run_id()
        summary = run_feature_extraction(upload_path=input_path, run_id=run_id, use_llm=getattr(args, "use_llm", False) or args.command == "run" and args.extract_nodes)
        print(f"Run complete: {run_id}")
        for key, value in summary.get("artifacts", {}).items():
            if value:
                print(f"- {key}: {value}")
        print(f"- features_file: data/runs/{run_id}/features.json")
        return

    if args.command == "create-questions":
        run_id = new_run_id()
        summary = run_question_generation_from_features_file(
            features_file=args.features_file,
            run_id=run_id,
            question_topic=args.question_topic,
            question_skeleton=args.question_skeleton,
            answer_type=args.answer_type,
            count=args.count,
            analysis_file=args.analysis_file,
            user_context=args.user_context,
        )
        print(json.dumps(summary, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
