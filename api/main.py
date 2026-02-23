import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .pipeline import (
    PipelineError,
    run_problem_set_generation_from_features_file,
    run_random_question_from_features_file,
    run_feature_extraction,
    run_pipeline,
    run_question_generation,
    run_question_generation_from_features_file,
)
from .storage import new_run_id, run_path, save_upload_file

app = FastAPI(title="SAT Question Gen API", version="1.0.0")
WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(WEB_DIR):
    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


class GenerateQuestionsRequest(BaseModel):
    feature_run_id: str = Field(..., description="Run ID from /extract-features")
    question_topic: str
    question_skeleton: str
    answer_type: str = Field(default="multiple_choice", pattern="^(multiple_choice|free_response)$")
    count: int = Field(default=1, ge=1, le=20)
    analysis_file: Optional[str] = Field(
        default=None,
        description="Optional path to a precomputed dataset analysis JSON file.",
    )
    user_context: Optional[str] = None


class CreateQuestionsRequest(BaseModel):
    features_file: str = Field(..., description="Path to features.json produced by /extract-nodes")
    question_topic: str
    question_skeleton: str
    answer_type: str = Field(default="multiple_choice", pattern="^(multiple_choice|free_response)$")
    count: int = Field(default=1, ge=1, le=20)
    analysis_file: Optional[str] = Field(
        default=None,
        description="Optional path to an analysis JSON. If omitted, uses analysis embedded in features_file.",
    )
    user_context: Optional[str] = None


class CreateRandomQuestionRequest(BaseModel):
    features_file: str = Field(..., description="Path to features.json produced by /extract-nodes")
    analysis_file: Optional[str] = None
    user_context: Optional[str] = None


class CreateProblemSetRequest(BaseModel):
    features_file: str = Field(..., description="Path to features.json produced by /extract-nodes")
    count: int = Field(default=10, ge=1, le=100)
    analysis_file: Optional[str] = None
    user_context: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def ui_home() -> HTMLResponse:
    ui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web", "index.html"))
    if not os.path.isfile(ui_path):
        return HTMLResponse("<h1>UI not found</h1>", status_code=404)
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/testing/default-features-file")
def get_default_features_file() -> JSONResponse:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    features_bundled_path = os.path.join(project_root, "test_data", "sat_questions.features.json")
    dataset_path = os.path.join(project_root, "test_data", "sat_questions.json")
    analysis_path = os.path.join(project_root, "test_data", "sat_questions.analysis.json")
    processed_dir = os.path.join(project_root, "data", "processed")
    runs_dir = os.path.join(project_root, "data", "runs")

    if not os.path.isfile(dataset_path):
        raise HTTPException(status_code=404, detail=f"Default dataset not found: {dataset_path}")

    if os.path.isfile(features_bundled_path):
        content = {
            "run_id": None,
            "features_file": "test_data/sat_questions.features.json",
            "dataset_file": "test_data/sat_questions.json",
            "analysis_file": "test_data/sat_questions.analysis.json" if os.path.isfile(analysis_path) else None,
            "cache_hit": True,
            "source": "bundled",
        }
        return JSONResponse(content=content)

    features_candidates = []
    if os.path.isdir(processed_dir):
        for name in os.listdir(processed_dir):
            candidate = os.path.join(processed_dir, name, "features.json")
            if os.path.isfile(candidate):
                features_candidates.append(candidate)

    if not features_candidates and os.path.isdir(runs_dir):
        for name in os.listdir(runs_dir):
            candidate = os.path.join(runs_dir, name, "features.json")
            if os.path.isfile(candidate):
                features_candidates.append(candidate)

    if not features_candidates:
        raise HTTPException(
            status_code=404,
            detail="No existing features.json found. Run /extract-nodes once to create one.",
        )

    features_candidates.sort(key=os.path.getmtime, reverse=True)
    selected = features_candidates[0]
    selected_rel = os.path.relpath(selected, project_root)
    content = {
        "run_id": None,
        "features_file": selected_rel.replace("\\", "/"),
        "dataset_file": "test_data/sat_questions.json",
        "analysis_file": "test_data/sat_questions.analysis.json" if os.path.isfile(analysis_path) else None,
        "cache_hit": True,
        "source": "local_fallback",
    }
    return JSONResponse(content=content)

def _run_from_upload(
    file: UploadFile,
    extract_nodes: bool,
    generate_count: int,
    user_id: Optional[str],
    memory_query: Optional[str],
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    run_id = new_run_id()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        if generate_count > 0:
            extract_nodes = True

        upload_path = save_upload_file(tmp_path, file.filename, run_id)
        summary = run_pipeline(
            upload_path=upload_path,
            run_id=run_id,
            extract_nodes=extract_nodes,
            generate_count=generate_count,
            user_id=user_id,
            memory_query=memory_query,
        )
        summary["upload_path"] = upload_path
        summary["run_dir"] = os.path.join("data", "runs", run_id)
        return JSONResponse(content=summary)
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _extract_features_from_upload(file: UploadFile, use_llm: bool) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    run_id = new_run_id()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        upload_path = save_upload_file(tmp_path, file.filename, run_id)
        summary = run_feature_extraction(upload_path=upload_path, run_id=run_id, use_llm=use_llm)
        summary["upload_path"] = upload_path
        summary["run_dir"] = os.path.join("data", "runs", run_id)
        return JSONResponse(content=summary)
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/process")
def process_dataset(
    file: UploadFile = File(...),
    extract_nodes: bool = Form(True),
    generate_count: int = Form(0),
    user_id: Optional[str] = Form(None),
    memory_query: Optional[str] = Form(None),
) -> JSONResponse:
    return _run_from_upload(file, extract_nodes, generate_count, user_id, memory_query)


@app.post("/extract-features")
def extract_features(
    file: UploadFile = File(...),
    use_llm: bool = Form(True),
) -> JSONResponse:
    return _extract_features_from_upload(file, use_llm)


@app.post("/extract-nodes")
def extract_nodes(
    file: UploadFile = File(...),
    use_llm: bool = Form(True),
) -> JSONResponse:
    return _extract_features_from_upload(file, use_llm)


@app.post("/generate-questions")
def generate_questions(request: GenerateQuestionsRequest) -> JSONResponse:
    run_id = new_run_id()
    try:
        summary = run_question_generation(
            feature_run_id=request.feature_run_id,
            run_id=run_id,
            question_topic=request.question_topic,
            question_skeleton=request.question_skeleton,
            answer_type=request.answer_type,
            count=request.count,
            analysis_file=request.analysis_file,
            user_context=request.user_context,
        )
        summary["run_dir"] = os.path.join("data", "runs", run_id)
        return JSONResponse(content=summary)
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Question generation failed: {exc}")


@app.post("/create-questions")
def create_questions(request: CreateQuestionsRequest) -> JSONResponse:
    run_id = new_run_id()
    try:
        summary = run_question_generation_from_features_file(
            features_file=request.features_file,
            run_id=run_id,
            question_topic=request.question_topic,
            question_skeleton=request.question_skeleton,
            answer_type=request.answer_type,
            count=request.count,
            analysis_file=request.analysis_file,
            user_context=request.user_context,
        )
        summary["run_dir"] = os.path.join("data", "runs", run_id)
        return JSONResponse(content=summary)
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Create questions failed: {exc}")


@app.post("/create-random-question")
def create_random_question(request: CreateRandomQuestionRequest) -> JSONResponse:
    run_id = new_run_id()
    try:
        summary = run_random_question_from_features_file(
            features_file=request.features_file,
            run_id=run_id,
            analysis_file=request.analysis_file,
            user_context=request.user_context,
        )
        summary["run_dir"] = os.path.join("data", "runs", run_id)
        return JSONResponse(content=summary)
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Create random question failed: {exc}")


@app.post("/create-question-problem-set")
def create_question_problem_set(request: CreateProblemSetRequest) -> JSONResponse:
    run_id = new_run_id()
    try:
        summary = run_problem_set_generation_from_features_file(
            features_file=request.features_file,
            run_id=run_id,
            count=request.count,
            analysis_file=request.analysis_file,
            user_context=request.user_context,
        )
        summary["run_dir"] = os.path.join("data", "runs", run_id)
        return JSONResponse(content=summary)
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Create problem set failed: {exc}")


@app.post("/runs")
def create_run_compat(
    file: UploadFile = File(...),
    extract_nodes: bool = Form(True),
    generate_count: int = Form(0),
    user_id: Optional[str] = Form(None),
    memory_query: Optional[str] = Form(None),
) -> JSONResponse:
    return _run_from_upload(file, extract_nodes, generate_count, user_id, memory_query)


@app.get("/feature-runs/{feature_run_id}/features-file")
def get_features_file(feature_run_id: str) -> JSONResponse:
    source_dir = run_path(feature_run_id)
    features_file = os.path.join(source_dir, "features.json")
    if not os.path.isdir(source_dir):
        raise HTTPException(status_code=404, detail=f"Feature run not found: {feature_run_id}")
    if not os.path.isfile(features_file):
        raise HTTPException(status_code=404, detail=f"features.json not found for run: {feature_run_id}")

    return JSONResponse(
        content={
            "feature_run_id": feature_run_id,
            "features_file": features_file.replace("\\", "/"),
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)
