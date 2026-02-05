from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import io
import uuid
import os
import re
from typing import Dict, Any

import numpy as np

from db import connect, disconnect
from log import log_file_upload
from fastapi.encoders import jsonable_encoder

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

_NAME_PATTERN = re.compile(r"^[A-Za-z]+(?:\s[A-Za-z]+)*$")

# Updated email regex for data quality checks
_EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9._%+-]+@"        # local part (letters, digits, ., _, %, +, -)
    r"[A-Za-z0-9.-]+\."           # domain + dot
    r"[A-Za-z]{2,}$"              # top-level domain (at least 2 letters)
)


_DATE_PATTERN = re.compile(
    r"^("
    r"\d{4}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])|"        # YYYY-MM-DD or YYYY/MM/DD
    r"(0[1-9]|[12]\d|3[01])[-/](0[1-9]|1[0-2])[-/]\d{4}|"        # DD-MM-YYYY or DD/MM/YYYY
    r"(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/]\d{4}"         # MM-DD-YYYY or MM/DD/YYYY
    r")$"
)

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{4,}$")



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


def validate_series(series: pd.Series, rule: str) -> tuple[int, int]:
    s = series.fillna("").astype(str).str.strip()  # replace NaN with ""
    if s.empty:
        return 0, 0

    if rule == "email":
        valid_mask = s.str.fullmatch(_EMAIL_PATTERN).fillna(False)
    elif rule == "name":
        valid_mask = s.str.fullmatch(_NAME_PATTERN).fillna(False)
    elif rule == "id":
        valid_mask = s.str.fullmatch(_IDENTIFIER_PATTERN).fillna(False)
    else:
        return 0, 0

    valid = int(valid_mask.sum())
    invalid = int(len(s) - valid)  # total minus valid = invalid
    return valid, invalid

def compute_type_consistency(df: pd.DataFrame) -> dict:
    type_consistency = {}
    rows = len(df)

    for col in df.columns:
        s = df[col].fillna("").astype(str).str.strip()  # <-- fillna here too

        missing = int((s == "").sum())
        non_missing = rows - missing

        rule = infer_column_rule(col)

        valid = invalid = 0

        if rule:
            valid, invalid = validate_series(s, rule)
        else:
            # For non-rule columns, treat all non-empty rows as valid
            valid = non_missing
            invalid = 0

        type_consistency[col] = {
            "missing": missing,
            "valid": valid,
            "invalid": invalid,
            "total": non_missing,
            "rule": rule or "none",
        }

    return type_consistency


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


def _compute_string_length_metrics(series: pd.Series) -> Dict[str, float]:
    """
    Compute basic length statistics for a text-like series.
    Returns an empty dict if there are no non-null string values.
    """
    s = series.dropna().astype(str)
    if s.empty:
        return {}

    lengths = s.str.len()
    if lengths.empty:
        return {}

    return {
        "min": float(lengths.min()),
        "max": float(lengths.max()),
        "avg": float(lengths.mean()),
    }


def infer_column_rule(col_name: str) -> str | None:
    """
    Infers the type rule for a column based on its name.
    Supports more variations for email, name, and id columns.
    """
    name = col_name.lower()

    # ---- email variations ----
    email_keywords = ["email", "e-mail", "mail_address", "email_address"]
    if any(k in name for k in email_keywords):
        return "email"

    # ---- name variations ----
    name_keywords = ["name", "full_name", "first_name", "last_name", "username", "user_name"]
    if any(k in name for k in name_keywords):
        return "name"

    # ---- id variations ----
    id_keywords = ["id", "_id", "user_id", "employee_id", "uuid", "identifier"]
    if any(k in name for k in id_keywords):
        return "id"

    return None



def _regex_confidence(series: pd.Series, pattern: re.Pattern) -> float:
    """
    Compute the fraction (0–1) of non-null, non-empty string values
    that fully match the given compiled regex pattern.
    """
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    total = len(s)
    if total == 0:
        return 0.0

    matches = s.str.fullmatch(pattern).fillna(False)
    return float(matches.sum() / total)


def _detect_string_formats(series: pd.Series) -> Dict[str, float]:
    """
    Detect common string formats and return confidence ratios.
    """
    email_conf = _regex_confidence(series, _EMAIL_PATTERN)
    date_conf = _regex_confidence(series, _DATE_PATTERN)
    identifier_conf = _regex_confidence(series, _IDENTIFIER_PATTERN)

    return {
        "email": email_conf,
        "date": date_conf,
        "identifier": identifier_conf,
    }

def analyze_columns(df: pd.DataFrame, numeric_df: pd.DataFrame, type_consistency: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = len(df)
    results: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        s = df[col]
        col_result: Dict[str, Any] = {}

        # ---- Determine type check ----
        dominant_type, type_counts = _infer_column_type_probabilities(s)
        col_result["type_check"] = dominant_type

        # ---- Update type_consistency for columns with rules ----
        rule = infer_column_rule(col)
        if rule and col in type_consistency:
            # Use validate_series directly for valid/invalid counts
            valid, invalid = validate_series(s, rule)
            type_consistency[col]["valid"] = valid
            type_consistency[col]["invalid"] = invalid

        # ---- inferred data type for numeric calculations ----
        if pd.api.types.is_numeric_dtype(numeric_df[col]):
            inferred_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            inferred_type = "datetime"
        else:
            inferred_type = "categorical"
        col_result["inferred_type"] = inferred_type

        # ---- missing values ----
        missing_count = int(s.isna().sum())
        col_result["missing_pct"] = (missing_count / rows) * 100 if rows else 0.0

        # ---- cardinality ----
        col_result["cardinality"] = int(s.nunique(dropna=True))

        # ---- numeric metrics ----
        num = numeric_df[col].dropna()
        if not num.empty and inferred_type == "numeric":
            col_result.update({
                "min": float(num.min()),
                "max": float(num.max()),
                "mean": float(num.mean()),
                "count": int(num.count()),
                "median": float(num.median()),
                "std": float(num.std(ddof=0)),
                "distribution": _numeric_distribution_metrics(num),
            })

        # ---- string/text metrics ----
        if inferred_type == "categorical":
            string_length = _compute_string_length_metrics(s)
            if string_length:
                col_result["string_length"] = string_length

        results[col] = col_result

    return results



def analyze_csv(content: bytes) -> Dict[str, Any]:
    df = load_dataframe(content)

    if df.empty:
        return {
            "cleaned_data": {"rows": 0, "columns": 0},
            "quality_report": {},
        }

    rows, cols = df.shape

    missing_rate, duplicate_rate, duplicate_count = compute_basic_quality(df)
    
    # Identify columns that should NOT be numeric
    text_columns = [col for col in df.columns if infer_column_rule(col) in ("name", "email", "id")]

    # Convert only non-text columns
    numeric_df = df.copy()
    for col in df.columns:
        if col not in text_columns:
            numeric_df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            numeric_df[col] = pd.NA  # prevent text columns from being treated as numeric


    type_consistency = compute_type_consistency(df)
    column_analysis = analyze_columns(df, numeric_df,type_consistency)
    outlier_rate = compute_outlier_rate(numeric_df)
    uniqueness, numeric_stats = compute_column_stats(df,numeric_df)
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
            "column_analysis": column_analysis,
        },
    }


@app.post("/analyze/{job_id}")
async def analyze(job_id: str):
    content = uploaded_files.get(job_id)
    if content is None:
        jobs[job_id] = "failed"
        return {"detail": "No uploaded file found for this job."}

    jobs[job_id] = "running"
    raw_results = analyze_csv(content)
    results_store[job_id] = to_native(raw_results) 
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


# NEW helper for numeric distribution analysis
def _numeric_distribution_metrics(series: pd.Series, bins: int = 10, sample_size: int = 100000) -> Dict[str, Any]:
    """
    Compute histogram, quantiles, skewness, kurtosis, zero-inflation, and detect constant/near-constant columns.
    Uses sampling for very large series.
    """
    s = series.dropna()
    if s.empty:
        return {}

    # Sample if very large
    if len(s) > sample_size:
        s = s.sample(sample_size, random_state=42)

    metrics: Dict[str, Any] = {}

    # histogram counts
    counts, bin_edges = np.histogram(s, bins=bins)
    metrics["histogram"] = {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist()
    }

    # quantiles
    quantiles = s.quantile([0.25, 0.5, 0.75, 0.95])
    metrics["quantiles"] = {str(q): float(v) for q, v in quantiles.items()}

    # skewness & kurtosis
    metrics["skewness"] = float(s.skew())
    metrics["kurtosis"] = float(s.kurt())

    # zero inflation
    metrics["zero_ratio"] = float((s == 0).sum() / len(s))

    # constant or near-constant detection
    metrics["is_constant"] = bool(s.nunique() <= 1)
    metrics["is_near_constant"] = bool(s.nunique() / len(s) < 0.01)  # <1% unique

    return metrics



def to_native(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def _infer_column_type_probabilities(series: pd.Series) -> tuple[str, Dict[str, int]]:
    """
    Infers the dominant type of a column and counts per-type occurrences.
    """
    counts: Dict[str, int] = {"int": 0, "float": 0, "string": 0, "datetime": 0, "bool": 0, "other": 0}

    for val in series.dropna():
        # bool check first
        if isinstance(val, bool):
            counts["bool"] += 1
        # numeric
        elif isinstance(val, int):
            counts["int"] += 1
        elif isinstance(val, float):
            if val.is_integer():
                counts["int"] += 1
            else:
                counts["float"] += 1
        # string: try to parse as int, float, datetime
        elif isinstance(val, str):
            val = val.strip()
            if val == "":
                continue
            # try int
            try:
                v_int = int(val)
                counts["int"] += 1
                continue
            except:
                pass
            # try float
            try:
                v_float = float(val)
                counts["float"] += 1
                continue
            except:
                pass
            # try datetime
            try:
                pd.to_datetime(val)
                counts["datetime"] += 1
                continue
            except:
                pass
            # otherwise string
            counts["string"] += 1
        else:
            counts["other"] += 1

    # pick dominant type
    dominant_type = max(counts.items(), key=lambda x: x[1])[0]
    return dominant_type, counts

    # pick dominant type
    dominant_type = max(counts.items(), key=lambda x: x[1])[0]
    return dominant_type, counts

