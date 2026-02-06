import io
import os
import uuid
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from fastapi import APIRouter, UploadFile, File, BackgroundTasks

from analysis_service import analyze_csv, analyze_csv_streaming, to_native
from log import log_file_upload
from state import jobs, uploaded_files, job_modes, results_store


router = APIRouter()
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"


@router.post("/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, str]:
    file_name = file.filename
    extension = os.path.splitext(file_name)[1].lstrip(".").lower() or "unknown"

    job_id = str(uuid.uuid4())
    dest_path = UPLOAD_DIR / f"{job_id}.csv"
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    # Stream the uploaded file to disk in chunks (e.g. 1 MB)
    with dest_path.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            total_bytes += len(chunk)

    await log_file_upload(
        user_name="anonymous",
        action="upload_file",
        file_name=file_name,
        extension=extension,
        file_size=total_bytes,
    )

    # File size is used ONLY for classification, never for validation.
    # The system is designed for massive datasets (hundreds of GBs). Every file
    # is accepted and stored; no hard or soft size limits are enforced.
    file_size = os.path.getsize(dest_path)
    LARGE_FILE_THRESHOLD_BYTES = 100 * 1024 * 1024  # 100 MB

    # Store the file path and mark mode for analysis strategy selection.
    uploaded_files[job_id] = str(dest_path)
    jobs[job_id] = "starting"
    job_modes[job_id] = "large" if file_size > LARGE_FILE_THRESHOLD_BYTES else "normal"

    return {"job_id": job_id}


MAX_IN_MEMORY_BYTES = 100 * 1024 * 1024  # 100 MB threshold


@router.post("/analyze/{job_id}")
async def analyze(job_id: str, background_tasks: BackgroundTasks) -> Dict[str, str] | Dict[str, str]:
    data_ref = uploaded_files.get(job_id)
    if data_ref is None:
        jobs[job_id] = "failed"
        return {"detail": "No uploaded file found for this job."}

    jobs[job_id] = "running"

    def run_analysis(job_id: str, data_ref: Any) -> None:
        try:
            # File size / mode are used only to adapt analysis strategy, never to reject.
            # The system is designed for massive datasets (hundreds of GBs).
            if isinstance(data_ref, str):
                mode = job_modes.get(job_id)
                try:
                    file_size = os.path.getsize(data_ref)
                except OSError:
                    jobs[job_id] = "failed"
                    return

                # For "large" mode or files > in-memory threshold: prefer streaming.
                # Streaming skips or approximates expensive metrics (exact duplicates,
                # full uniqueness, heavy outlier detection) to keep memory and CPU
                # under control for very large files.
                if mode == "large" or file_size > MAX_IN_MEMORY_BYTES:
                    # Large file: use streaming analysis to avoid excessive memory use
                    raw_results = analyze_csv_streaming(data_ref)
                else:
                    # Small enough to load fully for maximum detail
                    with open(data_ref, "rb") as f:
                        content = f.read()
                    raw_results = analyze_csv(content)
            else:
                # Legacy: data_ref already holds bytes in memory
                raw_results = analyze_csv(data_ref)

            results_store[job_id] = to_native(raw_results)
            jobs[job_id] = "completed"
        except Exception:
            jobs[job_id] = "failed"

    background_tasks.add_task(run_analysis, job_id, data_ref)

    return {"message": "analysis started"}


@router.get("/status/{job_id}")
async def status(job_id: str) -> Dict[str, str]:
    return {"status": jobs.get(job_id, "unknown")}


@router.get("/results/{job_id}")
async def results(job_id: str) -> Dict[str, Any]:
    data = results_store.get(job_id)
    if not data:
        return {"detail": "Results not found"}
    return data


@router.get("/raw/{job_id}")
async def raw_data(job_id: str) -> Dict[str, Any]:
    data_ref = uploaded_files.get(job_id)
    if data_ref is None:
        return {"detail": "No data found for this job."}

    # Backwards-compatible: handle both bytes (old) and file path (new)
    if isinstance(data_ref, str):
        df = pd.read_csv(data_ref)
    else:
        df = pd.read_csv(io.BytesIO(data_ref))

    # Replace all NaN / Inf with None
    df = df.replace({pd.NA: None, float("nan"): None, float("inf"): None, float("-inf"): None})
    df = df.where(pd.notna(df), None)  # extra safety

    return {
        "header": df.columns.tolist(),
        "rows": df.values.tolist(),  # NO astype(str)
    }

