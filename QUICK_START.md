# SAT Question Generator — Quick Start

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env.local` in the project root:

```env
GEMINI_API_KEY=your_key_here
# Optional:
# SUPERMEMORY_API_KEY=your_key_here
```

## 2) Validate Dataset (No API Calls)

```bash
python -m api.cli --input test_data/comprehensive_dataset.json --dry-run
```

## 3) Run Pipeline from CLI

Extract graph nodes only:

```bash
python -m api.cli --input test_data/comprehensive_dataset.json --extract-nodes
```

Extract nodes and generate questions:

```bash
python -m api.cli --input test_data/comprehensive_dataset.json --extract-nodes --generate-count 5
```

Artifacts are written to `data/runs/run_<timestamp>_<id>/`.

## 4) Run API Server

```bash
uvicorn api.main:app --reload
```

Available endpoints:

- `GET /health`
- `POST /process` (multipart form: `file`, optional `extract_nodes`, `generate_count`, `user_id`, `memory_query`)

Example request:

```bash
curl -X POST "http://localhost:8000/process" ^
  -F "file=@test_data/comprehensive_dataset.json" ^
  -F "extract_nodes=true" ^
  -F "generate_count=2"
```

## 5) Run Tests

```bash
python test_pipeline_mocked.py
python test_api_server.py
```
