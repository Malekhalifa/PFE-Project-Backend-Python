from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import csv
import io
import math
import uuid
from typing import Dict, List, Any
from db import connect, disconnect
from log import log_file_upload
import json
import os
app = FastAPI()


@app.on_event("startup")
async def startup():
    await connect()

@app.on_event("shutdown")
async def shutdown():
    await disconnect()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In‑memory storage for uploaded files and job results
jobs: Dict[str, str] = {}  # job_id -> status
uploaded_files: Dict[str, bytes] = {}  # job_id -> raw CSV bytes
results_store: Dict[str, Dict[str, Any]] = {}  # job_id -> analysis result

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()

    file_name = file.filename
    extension = os.path.splitext(file_name)[1].lstrip(".").lower() or "unknown"
    file_size = len(content)

    await log_file_upload(
        user_name="anonymous",
        action="upload_file",
        file_name=file_name,
        extension=extension,
        file_size=file_size,
    )
    job_id = str(uuid.uuid4())
    uploaded_files[job_id] = content
    jobs[job_id] = "starting"
    return {"job_id": job_id}



def _parse_csv(content: bytes) -> List[List[str]]:
    text = content.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows: List[List[str]] = [row for row in reader]
    return rows


def _is_missing(value: str) -> bool:
    v = value.strip()
    return v == "" or v.lower() in {"na", "null", "none", "nan"}


def _is_valid_email(value: str) -> bool:
    """
    Very simple email heuristic:
    - must contain exactly one "@"
    - must have at least one "." in the domain part
    """
    value = value.strip()
    if "@" not in value:
        return False
    local, _, domain = value.partition("@")
    if not local or not domain:
        return False
    return "." in domain


def _compute_outliers(numeric_columns: List[List[float]]) -> int:
    """
    Very simple outlier detection:
    A value is an outlier if it's more than 3 standard deviations from
    the mean of its column.
    """
    outliers = 0
    for col in numeric_columns:
        if len(col) < 2:
            continue
        mean = sum(col) / len(col)
        var = sum((x - mean) ** 2 for x in col) / (len(col) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        if std == 0:
            continue
        for x in col:
            if abs(x - mean) > 3 * std:
                outliers += 1
    return outliers


def analyze_csv(content: bytes) -> Dict[str, Any]:
    rows = _parse_csv(content)
    if not rows:
        return {
            "cleaned_data": {"rows": 0, "columns": 0},
            "quality_report": {},
        }

    header = rows[0]
    data_rows = rows[1:] if len(rows) > 1 else []

    num_rows = len(data_rows)
    num_cols = len(header)

    # --- Missing values ---
    total_cells = num_rows * num_cols if num_rows > 0 else 0
    missing_count = 0
    for row in data_rows:
        for cell in row:
            if _is_missing(cell):
                missing_count += 1
    missing_rate = missing_count / total_cells if total_cells > 0 else 0.0

    # --- Duplicate rows ---
    row_tuples = [tuple(r) for r in data_rows]
    unique_rows = set(row_tuples)
    duplicate_count = len(row_tuples) - len(unique_rows)
    duplicate_rate = duplicate_count / num_rows if num_rows > 0 else 0.0

    # --- Numeric processing & type consistency ---
    numeric_columns: List[List[float]] = [[] for _ in range(num_cols)]
    type_consistency: Dict[str, Dict[str, int]] = {}
    email_validity: Dict[str, Dict[str, int]] = {}
    for idx, col_name in enumerate(header):
        # Base counters for every column
        type_consistency[col_name] = {
            "numeric": 0,
            "non_numeric": 0,
            "missing": 0,
            "valid": 0,
            "invalid": 0,
            "total": 0,
        }
        # Track which columns we consider "email-like"
        if "email" in col_name.lower():
            email_validity[col_name] = {"valid": 0, "invalid": 0, "total": 0}

    for row in data_rows:
        for idx in range(num_cols):
            if idx >= len(row):
                continue
            cell = row[idx].strip()
            col_name = header[idx]

            if _is_missing(cell):
                type_consistency[col_name]["missing"] += 1
                continue

            # Email validity check for columns that look like emails
            if col_name in email_validity:
                email_validity[col_name]["total"] += 1
                type_consistency[col_name]["total"] += 1
                if _is_valid_email(cell):
                    email_validity[col_name]["valid"] += 1
                    type_consistency[col_name]["valid"] += 1
                else:
                    email_validity[col_name]["invalid"] += 1
                    type_consistency[col_name]["invalid"] += 1

            try:
                value = float(cell)
                numeric_columns[idx].append(value)
                type_consistency[col_name]["numeric"] += 1
            except ValueError:
                type_consistency[col_name]["non_numeric"] += 1

    # After scanning all rows, ensure "total" is at least numeric + non_numeric
    for col_name, counts in type_consistency.items():
        counts["total"] = max(
            counts.get("total", 0),
            counts.get("numeric", 0) + counts.get("non_numeric", 0),
        )

    # --- Outliers ---
    outlier_count = _compute_outliers(numeric_columns)
    numeric_cell_count = sum(len(col) for col in numeric_columns)
    outlier_rate = (
        outlier_count / numeric_cell_count if numeric_cell_count > 0 else 0.0
    )

    # --- Uniqueness per column ---
    uniqueness: Dict[str, float] = {}
    for idx, col_name in enumerate(header):
        col_values = [row[idx] for row in data_rows if idx < len(row)]
        unique_count = len(set(col_values))
        uniqueness[col_name] = unique_count / len(col_values) if col_values else 0.0

    # --- Basic stats per numeric column ---
    numeric_stats: Dict[str, Dict[str, float]] = {}
    for idx, col_name in enumerate(header):
        col = numeric_columns[idx]
        if col:
            mean = sum(col) / len(col)
            var = sum((x - mean) ** 2 for x in col) / (len(col) - 1) if len(col) > 1 else 0.0
            std = math.sqrt(var)
            numeric_stats[col_name] = {
                "min": min(col),
                "max": max(col),
                "mean": mean,
            }
        else:
            numeric_stats[col_name] = {}

   

    # --- Quality score ---
    penalty = (missing_rate + duplicate_rate + outlier_rate) / 3.0
    quality_score = max(0.0, 1.0 - penalty)

    return {
        "cleaned_data": {"rows": num_rows, "columns": num_cols},
        "quality_report": {
            "missing_rate": round(missing_rate, 4),
            "duplicate_rate": round(duplicate_rate, 4),
            "outlier_rate": round(outlier_rate, 4),
            "duplicate_count": duplicate_count,
            "quality_score": round(quality_score, 4),
            "uniqueness": {k: round(v, 4) for k, v in uniqueness.items()},
            "type_consistency": type_consistency,
            "numeric_stats": numeric_stats,
            "email_validity": email_validity,
        },
    }


@app.post("/analyze/{job_id}")
async def analyze(job_id: str):
    """
    Perform deterministic, static analysis on the uploaded CSV for this job.
    """
    content = uploaded_files.get(job_id)
    if content is None:
        jobs[job_id] = "failed"
        return {"detail": "No uploaded file found for this job."}

    jobs[job_id] = "running"
    result = analyze_csv(content)
    results_store[job_id] = result
    jobs[job_id] = "completed"

    return {"message": "analysis completed"}


@app.get("/status/{job_id}")
async def status(job_id: str):
    status_value = jobs.get(job_id, "unknown")
    return {"status": status_value}


@app.get("/results/{job_id}")
async def results(job_id: str):
    data = results_store.get(job_id)
    if not data:
        return {"detail": "Results not found"}
    return data


@app.get("/raw/{job_id}")
async def raw_data(job_id: str):
    """
    Return the cleaned CSV for this job as header + rows
    so the frontend can render a table matching the CSV format.
    """
    content = uploaded_files.get(job_id)
    if content is None:
        return {"detail": "No data found for this job."}

    rows = _parse_csv(content)
    if not rows:
        return {"header": [], "rows": []}

    header = rows[0]
    data_rows = rows[1:] if len(rows) > 1 else []
    return {"header": header, "rows": data_rows}