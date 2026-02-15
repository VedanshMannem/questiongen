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

## 3) Run Pipeline from CLI

Extract graph nodes only:

```bash
python -m api.cli --input test_data/sat_questions.json --extract-nodes
```

Extract nodes and generate questions:

```bash
python -m api.cli --input test_data/sat_questions.json --extract-nodes --generate-count 5
```

Artifacts are written to `data/runs/run_<timestamp>_<id>/`.

## 4) Run API Server

```bash
uvicorn api.main:app --reload
```

Available endpoints:

- `GET /health`
- `POST /extract-features` (multipart form: `file`, optional `use_llm`)
- `POST /generate-questions` (json body: `feature_run_id`, `question_topic`, `question_skeleton`, `answer_type`, `count`)
- `POST /process` (multipart form: `file`, optional `extract_nodes`, `generate_count`, `user_id`, `memory_query`)

Example request:

```bash
curl -X POST "http://localhost:8000/extract-features" ^
  -F "file=@test_data/comprehensive_dataset.json" ^
  -F "use_llm=true"
```

Then use the returned `run_id` as `feature_run_id`:

```bash
curl -X POST "http://localhost:8000/generate-questions" ^
  -H "Content-Type: application/json" ^
  -d "{\"feature_run_id\":\"run_20260215_033530_d9d637f0\",\"question_topic\":\"Information and Ideas / Inferences\",\"question_skeleton\":\"logical_completion\",\"answer_type\":\"multiple_choice\",\"count\":2}"
```

## 5) Run Tests

```bash
python test_api_server.py
```
