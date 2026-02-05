from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import io
import uuid
import os
from typing import Dict, Any

from db import connect, disconnect
from log import log_file_upload

import pandas as pd
from sklearn.ensemble import IsolationForest

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

# In-memory state (unchanged)
jobs: Dict[str, str] = {}
uploaded_files: Dict[str, bytes] = {}
results_store: Dict[str, Dict[str, Any]] = {}


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


def _is_valid_email_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return (
        s.str.contains("@", regex=False)
        & s.str.split("@").str.len().eq(2)
        & s.str.split("@").str[1].str.contains(".", regex=False)
    )

def load_dataframe(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(content))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def compute_basic_quality(df: pd.DataFrame) -> tuple[float, float, int]:
    rows, cols = df.shape
    total = rows * cols

    missing_rate = df.isna().sum().sum() / total if total else 0.0
    duplicate_count = int(df.duplicated().sum())
    duplicate_rate = duplicate_count / rows if rows else 0.0

    return missing_rate, duplicate_rate, duplicate_count

def compute_type_consistency(df: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    type_consistency = {}
    email_validity = {}

    rows = len(df)

    for col in df.columns:
        col_series = df[col]
        numeric_series = numeric_df[col]

        missing = int(col_series.isna().sum())
        numeric = int(numeric_series.notna().sum())
        non_numeric = rows - numeric - missing

        type_consistency[col] = {
            "numeric": numeric,
            "non_numeric": non_numeric,
            "missing": missing,
            "valid": 0,
            "invalid": 0,
            "total": numeric + non_numeric,
        }

        if "email" in col.lower():
            valid_mask = _is_valid_email_series(col_series.dropna())
            valid = int(valid_mask.sum())
            invalid = int((~valid_mask).sum())
            email_validity[col] = {
                "valid": valid,
                "invalid": invalid,
                "total": valid + invalid,
            }
            type_consistency[col]["valid"] = valid
            type_consistency[col]["invalid"] = invalid

    return type_consistency, email_validity, numeric_df

def compute_outlier_rate(numeric_df: pd.DataFrame) -> float:
    numeric_only = numeric_df.dropna(axis=1, how="all")

    if numeric_only.shape[0] <= 10 or numeric_only.shape[1] == 0:
        return 0.0

    filled = numeric_only.fillna(numeric_only.mean())
    if filled.shape[1] > 50:
        filled = filled.iloc[:, :50]

    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    preds = iso.fit_predict(filled)

    return (preds == -1).sum() / filled.shape[0]

def compute_column_stats(df: pd.DataFrame, numeric_df: pd.DataFrame) -> tuple[dict, dict]:
    rows = len(df)

    uniqueness = {
        col: df[col].nunique(dropna=False) / rows if rows else 0.0
        for col in df.columns
    }

    numeric_stats = {}
    for col in df.columns:
        col_num = numeric_df[col].dropna()
        numeric_stats[col] = {} if col_num.empty else {
            "min": float(col_num.min()),
            "max": float(col_num.max()),
            "mean": float(col_num.mean()),
        }

    return uniqueness, numeric_stats

def analyze_csv(content: bytes) -> Dict[str, Any]:
    df = load_dataframe(content)

    if df.empty:
        return {
            "cleaned_data": {"rows": 0, "columns": 0},
            "quality_report": {},
        }

    rows, cols = df.shape

    missing_rate, duplicate_rate, duplicate_count = compute_basic_quality(df)
    type_consistency, email_validity, numeric_df = compute_type_consistency(df)
    outlier_rate = compute_outlier_rate(numeric_df)
    uniqueness, numeric_stats = compute_column_stats(df, numeric_df)

    penalty = (missing_rate + duplicate_rate + outlier_rate) / 3.0
    quality_score = max(0.0, 1.0 - penalty)

    return {
        "cleaned_data": {"rows": rows, "columns": cols},
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
    content = uploaded_files.get(job_id)
    if content is None:
        jobs[job_id] = "failed"
        return {"detail": "No uploaded file found for this job."}

    jobs[job_id] = "running"
    results_store[job_id] = analyze_csv(content)
    jobs[job_id] = "completed"

    return {"message": "analysis completed"}


@app.get("/status/{job_id}")
async def status(job_id: str):
    return {"status": jobs.get(job_id, "unknown")}


@app.get("/results/{job_id}")
async def results(job_id: str):
    data = results_store.get(job_id)
    if not data:
        return {"detail": "Results not found"}
    return data


@app.get("/raw/{job_id}")
async def raw_data(job_id: str):
    content = uploaded_files.get(job_id)
    if content is None:
        return {"detail": "No data found for this job."}

    df = pd.read_csv(io.BytesIO(content))

    # Replace all NaN / Inf with None
    df = df.replace({pd.NA: None, float("nan"): None, float("inf"): None, float("-inf"): None})
    df = df.where(pd.notna(df), None)  # extra safety

    return {
        "header": df.columns.tolist(),
        "rows": df.values.tolist(),  # NO astype(str)
    }
