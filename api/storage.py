import json
import os
import shutil
import uuid
from datetime import datetime

def _workspace_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def data_dir() -> str:
    return os.path.join(_workspace_root(), "data")

def runs_dir() -> str:
    return os.path.join(data_dir(), "runs")

def uploads_dir() -> str:
    return os.path.join(data_dir(), "uploads")

def ensure_dirs() -> None:
    os.makedirs(data_dir(), exist_ok=True)
    os.makedirs(runs_dir(), exist_ok=True)
    os.makedirs(uploads_dir(), exist_ok=True)

def new_run_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{uuid.uuid4().hex[:8]}"

def run_path(run_id: str) -> str:
    return os.path.join(runs_dir(), run_id)

def create_run_dir(run_id: str) -> str:
    ensure_dirs()
    path = run_path(run_id)
    os.makedirs(path, exist_ok=True)
    return path

def save_json(path: str, data: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def save_upload_file(tmp_path: str, filename: str, run_id: str) -> str:
    ensure_dirs()
    safe_name = filename.replace("..", "_")
    dest = os.path.join(uploads_dir(), f"{run_id}_{safe_name}")
    shutil.copyfile(tmp_path, dest)
    return dest
