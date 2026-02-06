import io
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from fastapi import APIRouter, UploadFile, File, BackgroundTasks

from analysis_service import analyze_csv, analyze_csv_streaming, to_native
from log import log_file_upload
from state import jobs, uploaded_files, job_modes, analysis_meta, results_store


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
        started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        t0 = time.perf_counter()

        meta: Dict[str, Any] = {
            "started_at": started_at,
            "file_path": None,
            "file_size_bytes": None,
            "mode": job_modes.get(job_id, "normal"),
            "analysis_type": "in-memory",
        }

        try:
            # File size / mode are used only to adapt analysis strategy, never to reject.
            # The system is designed for massive datasets (hundreds of GBs).
            if isinstance(data_ref, str):
                meta["file_path"] = data_ref
                mode = job_modes.get(job_id)
                try:
                    file_size = os.path.getsize(data_ref)
                    meta["file_size_bytes"] = file_size
                except OSError:
                    jobs[job_id] = "failed"
                    return

                # For "large" mode or files > in-memory threshold: prefer streaming.
                # Streaming skips or approximates expensive metrics (exact duplicates,
                # full uniqueness, heavy outlier detection) to keep memory and CPU
                # under control for very large files.
                if mode == "large" or file_size > MAX_IN_MEMORY_BYTES:
                    meta["analysis_type"] = "streaming"
                    raw_results = analyze_csv_streaming(data_ref)
                else:
                    with open(data_ref, "rb") as f:
                        content = f.read()
                    raw_results = analyze_csv(content)
            else:
                raw_results = analyze_csv(data_ref)

            duration_ms = int((time.perf_counter() - t0) * 1000)
            meta["completed_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            meta["duration_ms"] = duration_ms
            analysis_meta[job_id] = meta

            results_store[job_id] = to_native(raw_results)
            jobs[job_id] = "completed"
        except Exception:
            jobs[job_id] = "failed"
            meta["completed_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            meta["duration_ms"] = int((time.perf_counter() - t0) * 1000)
            analysis_meta[job_id] = meta

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


def _build_export_report(job_id: str, data: Dict[str, Any], meta: Dict[str, Any] | None) -> Dict[str, Any]:
    """Build the downloadable report in the standardized export schema."""
    cleaned = data.get("cleaned_data") or {}
    qr = data.get("quality_report") or {}
    type_cons = qr.get("type_consistency") or {}
    num_stats = qr.get("numeric_stats") or {}
    col_analysis = qr.get("column_analysis") or {}
    uniqueness = qr.get("uniqueness") or {}

    duplicate_rate = qr.get("duplicate_rate")
    outlier_rate = qr.get("outlier_rate", 0)
    analysis_type = (meta or {}).get("analysis_type", "in-memory")
    is_streaming = analysis_type == "streaming"

    # Build column_analysis in export schema
    export_col_analysis: Dict[str, Any] = {}
    for col in set(list(type_cons.keys()) + list(col_analysis.keys()) + list(num_stats.keys())):
        tc = type_cons.get(col) or {}
        ca = col_analysis.get(col) or {}
        ns = num_stats.get(col) or {}
        rows = cleaned.get("rows") or 1
        missing_count = tc.get("missing", 0)
        missing_rate_col = (missing_count / rows) if rows else 0
        total = tc.get("total") or 0
        type_consistency = (tc.get("valid", 0) / total) if total else None

        issues: list[str] = []
        if missing_rate_col > 0.1:
            issues.append("high_missing_rate")
        if tc.get("invalid", 0) > 0 and total:
            issues.append("type_mismatch")

        export_col_analysis[col] = {
            "inferred_type": ca.get("type_check") or ca.get("inferred_type") or "string",
            "missing_count": missing_count,
            "missing_rate": round(missing_rate_col, 4),
            "type_consistency": round(type_consistency, 4) if type_consistency is not None else None,
            "numeric_stats": ns if ns else None,
            "uniqueness": {
                "unique_count": None,
                "unique_rate": uniqueness.get(col),
                "approximate": is_streaming,
            },
            "issues": issues,
        }

    limitations: Dict[str, Any] = {}
    if is_streaming:
        limitations["approximate_metrics"] = True
        limitations["skipped_checks"] = ["exact_duplicates", "full_uniqueness"]
    else:
        limitations["approximate_metrics"] = False
        limitations["skipped_checks"] = []

    return {
        "meta": {
            "job_id": job_id,
            "file_path": (meta or {}).get("file_path"),
            "file_size_bytes": (meta or {}).get("file_size_bytes"),
            "mode": (meta or {}).get("mode", "normal"),
            "analysis_type": analysis_type,
            "started_at": (meta or {}).get("started_at"),
            "completed_at": (meta or {}).get("completed_at"),
            "duration_ms": (meta or {}).get("duration_ms"),
            "version": "1.0",
        },
        "dataset_summary": {
            "rows": cleaned.get("rows", 0),
            "columns": cleaned.get("columns", 0),
        },
        "quality_overview": {
            "missing_rate": qr.get("missing_rate"),
            "duplicate_rate": duplicate_rate if duplicate_rate is not None else None,
            "error_rate": round(outlier_rate, 4) if isinstance(outlier_rate, (int, float)) else None,
        },
        "column_analysis": export_col_analysis,
        "limitations": limitations,
    }


@router.get("/results/{job_id}/export")
async def export_report(job_id: str) -> Dict[str, Any]:
    """Return the analysis report in the standardized export schema for download."""
    data = results_store.get(job_id)
    if not data:
        return {"detail": "Results not found"}
    meta = analysis_meta.get(job_id)
    return _build_export_report(job_id, data, meta)


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

