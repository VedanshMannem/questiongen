import argparse
import os
from typing import Optional

from .pipeline import run_pipeline, _load_questions, _normalize_question, _validate_questions, _compute_insights
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
    """Test run without making API calls - validates data and shows what would happen."""
    print("=" * 80)
    print("DRY RUN MODE - No API calls will be made")
    print("=" * 80)
    
    try:
        print("\n[1] Loading questions...")
        questions_raw = _load_questions(input_path)
        print(f"✓ Loaded {len(questions_raw)} questions")
        
        print("\n[2] Normalizing...")
        normalized = [_normalize_question(q, i) for i, q in enumerate(questions_raw)]
        _validate_questions(normalized)
        print(f"✓ Normalized and validated {len(normalized)} questions")
        
        print("\n[3] Computing insights...")
        insights = _compute_insights(normalized)
        print(f"✓ Insights computed")
        print(f"  - Avg question length: {insights['overall']['question_length']['average']} chars")
        print(f"  - Avg answer length: {insights['overall']['answer_length']['average']} chars")
        print(f"  - Domains: {len(insights['by_domain'])} unique")
        print(f"  - Skills per domain: {', '.join([str(len([s for s in insights['by_skill'] if s['domain']==d['domain']])) for d in insights['by_domain']])}")
        
        print("\n" + "=" * 80)
        print("DRY RUN SUCCESSFUL - Dataset is valid and ready for full pipeline")
        print("=" * 80)
        print("\nNext: python -m api.cli --input YOUR_FILE --extract-nodes --generate-count 5")
        
    except Exception as e:
        print(f"\n✗ Error in dry run: {e}")
        raise

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAT question pipeline on a dataset.")
    parser.add_argument("--input", help="Path to a JSON dataset file.")
    parser.add_argument("--input-folder", default=uploads_dir(), help="Folder to scan when --input is omitted.")
    parser.add_argument("--extract-nodes", action="store_true", help="Extract graph nodes.")
    parser.add_argument("--generate-count", type=int, default=0, help="Number of questions to generate.")
    parser.add_argument("--user-id", help="Reserved for future user-memory feature (currently disabled).")
    parser.add_argument("--memory-query", help="Reserved for future user-memory feature (currently disabled).")
    parser.add_argument("--dry-run", action="store_true", help="Validate data without making API calls.")

    args = parser.parse_args()

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
    summary = run_pipeline(
        upload_path=input_path,
        run_id=run_id,
        extract_nodes=args.extract_nodes or args.generate_count > 0,
        generate_count=args.generate_count,
        user_id=args.user_id,
        memory_query=args.memory_query,
    )

    print(f"Run complete: {run_id}")
    for key, value in summary.get("artifacts", {}).items():
        if value:
            print(f"- {key}: {value}")

if __name__ == "__main__":
    main()
