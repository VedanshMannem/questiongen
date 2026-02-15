import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .pipeline import PipelineError, run_pipeline
from .storage import new_run_id, save_upload_file

app = FastAPI(title="SAT Question Gen API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


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


@app.post("/process")
def process_dataset(
    file: UploadFile = File(...),
    extract_nodes: bool = Form(True),
    generate_count: int = Form(0),
    user_id: Optional[str] = Form(None),
    memory_query: Optional[str] = Form(None),
) -> JSONResponse:
    return _run_from_upload(file, extract_nodes, generate_count, user_id, memory_query)


@app.post("/runs")
def create_run_compat(
    file: UploadFile = File(...),
    extract_nodes: bool = Form(True),
    generate_count: int = Form(0),
    user_id: Optional[str] = Form(None),
    memory_query: Optional[str] = Form(None),
) -> JSONResponse:
    return _run_from_upload(file, extract_nodes, generate_count, user_id, memory_query)
