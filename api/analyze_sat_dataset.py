import argparse
import json
import os

from .dataset_analysis import analyze_json_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SAT question dataset for metadata, topics, and skeletons.")
    parser.add_argument("--input", default="test_data/sat_questions.json", help="Input SAT JSON file.")
    parser.add_argument(
        "--output",
        default="test_data/sat_questions.analysis.json",
        help="Output path for aggregated analysis JSON.",
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    analysis = analyze_json_file(input_path=input_path, output_path=output_path)

    print(f"Analyzed: {input_path}")
    print(f"Wrote: {output_path}")
    print(f"Total questions: {analysis.get('dataset_size', 0)}")

    top_domains = (analysis.get("metadata_summary", {}) or {}).get("domains", {})
    if top_domains:
        print("Top domains:")
        for name, count in list(top_domains.items())[:6]:
            print(f"- {name}: {count}")

    top_topics = ((analysis.get("core_topics", {}) or {}).get("groups", []) or [])[:8]
    if top_topics:
        print("Core topics:")
        for item in top_topics:
            print(f"- {item.get('topic')}: {item.get('count')}")


if __name__ == "__main__":
    main()
