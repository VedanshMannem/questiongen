# SAT Question Generator — Quick Start

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
GEMINI_API_KEY=your_key_here
```

## 2) Validate Dataset (No API Calls)

```bash
python -m api.cli --input test_data/comprehensive_dataset.json --dry-run
```

## 3) Build Dataset Analysis Artifacts

This generates grouped metadata, core topics, and reusable question skeleton families.

```bash
python -m api.analyze_sat_dataset --input test_data/sat_questions.json --output test_data/sat_questions.analysis.json
```

## 4) Run Pipeline from CLI

Extract nodes/features bundle (algorithmic + AI):

```bash
python -m api.cli extract-nodes --input test_data/sat_questions.json --use-llm
```

Create questions from the exported features file:

```bash
python -m api.cli create-questions --features-file data/runs/<RUN_ID>/features.json --question-topic "Information and Ideas / Inferences" --question-skeleton "A passage presents context and constraints, then leaves a blank. Select the choice that most logically completes the idea while preserving causal consistency." --answer-type multiple_choice --count 2 --analysis-file test_data/sat_questions.analysis.json
```

For deployment/demo without uploads, a pre-bundled features file is available:

```bash
test_data/sat_questions.features.json
```

Artifacts are written to `data/runs/run_<timestamp>_<id>/`.

## 5) Run API Server

```bash
uvicorn api.main:app --reload
```

Available endpoints:

- `GET /health`
- `GET /` (quick web UI for random/problem-set generation and answer checking)
- `POST /extract-nodes` (multipart form: `file`, optional `use_llm`)  ← primary endpoint
- `POST /create-questions` (json body: `features_file`, `question_topic`, `question_skeleton`, `answer_type`, `count`, optional `analysis_file`)  ← primary endpoint
- `POST /create-random-question` (json body: `features_file`, optional `analysis_file`, optional `user_context`)
- `POST /create-question-problem-set` (json body: `features_file`, `count`, optional `analysis_file`, optional `user_context`)
- `GET /feature-runs/{feature_run_id}/features-file` (returns canonical `features.json` path)
- `POST /extract-features` (compat)
- `POST /generate-questions` (compat)
- `POST /process` (multipart form: `file`, optional `extract_nodes`, `generate_count`, `user_id`, `memory_query`)

Example request:

```bash
curl -X POST "http://localhost:8000/extract-nodes" ^
  -F "file=@test_data/comprehensive_dataset.json" ^
  -F "use_llm=true"
```

Then create questions from the exported features bundle:

```bash
curl -X POST "http://localhost:8000/create-questions" ^
  -H "Content-Type: application/json" ^
  -d "{\"features_file\":\"data/runs/run_20260215_033530_d9d637f0/features.json\",\"question_topic\":\"Information and Ideas / Inferences\",\"question_skeleton\":\"A passage presents context and constraints, then leaves a blank. Select the choice that most logically completes the idea while preserving causal consistency.\",\"answer_type\":\"multiple_choice\",\"count\":2,\"analysis_file\":\"test_data/sat_questions.analysis.json\"}"
```

Generate one random question using weighted metadata distribution from the features bundle:

```bash
curl -X POST "http://localhost:8000/create-random-question" ^
  -H "Content-Type: application/json" ^
  -d "{\"features_file\":\"data/runs/run_20260215_033530_d9d637f0/features.json\",\"analysis_file\":\"test_data/sat_questions.analysis.json\"}"
```

Generate a problem set with distribution sampled from dataset metadata:

```bash
curl -X POST "http://localhost:8000/create-question-problem-set" ^
  -H "Content-Type: application/json" ^
  -d "{\"features_file\":\"data/runs/run_20260215_033530_d9d637f0/features.json\",\"count\":10,\"analysis_file\":\"test_data/sat_questions.analysis.json\"}"
```

`analysis_file` is optional. If provided, generation reuses that file every run and does not regenerate analysis.
If omitted, analysis is created once per loaded dataset and then reused from run artifacts.

## 6) Run Tests

```bash
python test.py
```
