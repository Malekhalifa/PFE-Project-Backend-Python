## Backend — FastAPI Service (`backend-python`)

This directory contains the **Python/FastAPI backend** for the project.  
It exposes a small REST API that lets the frontend upload a CSV file, trigger an analysis job, check job status, and fetch both the **quality report** and the **raw data**.

### High-level architecture

- **Framework**: FastAPI (served by `uvicorn`).
- **State**: All state is in memory (Python dicts) – **no database**.
- **Responsibilities split into modules**:
  - `main.py` – creates the FastAPI app, configures CORS, wires startup/shutdown events, and includes the API routes.
  - `routes.py` – defines all HTTP endpoints and how they use state and services.
  - `analysis_service.py` – pure data logic for CSV loading and quality analysis.
  - `state.py` – in-memory job/file/result stores.
  - `models.py` – Pydantic models for request/response schemas (type-safe responses).

Conceptually:

- The **frontend** calls the API.
- The **routes layer** orchestrates requests:
  - reads/writes from `state.py`
  - calls into `analysis_service.py`
  - returns Pydantic/JSON responses (defined in `models.py`).
- The **service layer** (`analysis_service.py`) is pure Python/pandas/numpy code with no FastAPI imports.

### File-by-file overview

- **`main.py`**
  - Creates the global `FastAPI` instance.
  - Registers `startup`/`shutdown` event handlers that call `connect()` / `disconnect()` from `db.py` (if you later add a real DB, this is where connections are managed).
  - Configures CORS to allow the Next.js dev server at `http://localhost:3000`.
  - Includes the router from `routes.py`:
    - `app.include_router(router)`

- **`routes.py`**
  - Defines the public API using an `APIRouter`:
    - `POST /upload` – accepts a CSV file, stores bytes in memory, logs the upload, and returns a `job_id`.
    - `POST /analyze/{job_id}` – runs the CSV analysis for a given job and stores the results.
    - `GET /status/{job_id}` – returns the job status (`starting`, `running`, `completed`, `failed`, or `unknown`).
    - `GET /results/{job_id}` – returns the computed quality report for the job.
    - `GET /raw/{job_id}` – returns the raw CSV header and rows for display as a table.
  - Imports:
    - `jobs`, `uploaded_files`, `results_store` from `state.py`.
    - `analyze_csv`, `to_native` from `analysis_service.py`.
    - `log_file_upload` from `log.py` (for basic audit logging).
    - Pydantic response models from `models.py` (if used).

- **`analysis_service.py`**
  - Contains **all CSV/data-quality logic**, independent from FastAPI:
    - `load_dataframe` – safely loads a `pandas.DataFrame` from uploaded bytes.
    - `compute_basic_quality` – missing-value rate, duplicate rate, duplicate count.
    - `compute_type_consistency`, `validate_series`, `infer_column_rule` – per-column rule checks (email/name/id patterns, etc.).
    - `compute_outlier_rate` – outlier detection over numeric columns using `IsolationForest`.
    - `compute_column_stats` – uniqueness and basic stats for each column.
    - `_numeric_distribution_metrics` – histogram, quantiles, skewness, kurtosis, zero inflation.
    - `_infer_column_type_probabilities` – heuristics for the dominant data type in a column.
    - `analyze_columns` – aggregates all per-column metrics.
    - `analyze_csv` – **main analysis entrypoint** used by routes:
      - Reads the CSV.
      - Builds numeric/text views.
      - Computes missing/duplicate/outlier metrics and a `quality_score`.
      - Returns a nested `dict` structure describing the dataset.
    - `to_native` – converts numpy/pandas scalar types to plain Python types so responses are JSON-serializable.

- **`state.py`**
  - Defines the **in-memory stores**:
    - `jobs: Dict[str, str]` – job status for each `job_id`.
    - `uploaded_files: Dict[str, bytes]` – raw CSV file contents per job.
    - `results_store: Dict[str, Dict[str, Any]]` – cached analysis results per job.
  - There is no persistence: restarting the backend clears everything.

- **`models.py`**
  - Pydantic models (if present) used to:
    - Document and type-check responses for the routes.
    - Make it easier to evolve the API while keeping types in sync.
  - Typical models (example, actual contents may differ):
    - `UploadResponse` – shape of `POST /upload` response.
    - `StatusResponse` – shape of `GET /status/{job_id}`.
    - `ResultsResponse` / `RawDataResponse` – shapes for results/raw endpoints.

- **`requirements.txt`**
  - Python dependencies for this backend (FastAPI, uvicorn, pandas, numpy, scikit-learn, etc.).
  - Install with:
    - `pip install -r requirements.txt`

### Request flow (end-to-end)

1. **Upload**
   - Frontend calls `POST /upload` with a CSV file.
   - Backend logs the upload, stores file bytes in `uploaded_files[job_id]`, initializes `jobs[job_id] = "starting"`, and returns `{"job_id": "<uuid>"}`.

2. **Analyze**
   - Frontend calls `POST /analyze/{job_id}`.
   - Route reads the stored file from `uploaded_files`.
   - Calls `analysis_service.analyze_csv(content)`.
   - Stores the result in `results_store[job_id]` and updates `jobs[job_id]` to `"completed"` (or `"failed"` if something goes wrong).

3. **Status polling**
   - Frontend periodically calls `GET /status/{job_id}` until it sees `"completed"` or `"failed"`.

4. **Results & raw data**
   - `GET /results/{job_id}` returns the quality report (metrics, per-column stats).
   - `GET /raw/{job_id}` returns:
     - `header`: list of column names.
     - `rows`: list-of-lists of row values (with nulls normalized to `null` in JSON).

### Large-file and streaming behavior

- **Uploads** are streamed to disk under `uploads/` (one file per job) instead of kept in memory. This avoids memory exhaustion for very large files (hundreds of GBs).

- **For large files**, analysis runs in **streaming mode** (`analyze_csv_streaming`), which processes the CSV in chunks. Some metrics are approximated or disabled (e.g. exact duplicate count, full uniqueness per column, complex outlier detection) to keep memory and CPU under control.

- **`POST /analyze/{job_id}`** returns immediately with `{"message": "analysis started"}`. The actual analysis runs in a **background task**. Use `GET /status/{job_id}` to poll until status is `completed` or `failed`, then `GET /results/{job_id}` to fetch the report.

### Running the backend locally

- From the `backend-python` directory:

```bash
pip install -r requirements.txt  # once
uvicorn main:app --reload --port 8000
```

- The frontend (Next.js) then talks to `http://127.0.0.1:8000` using these endpoints.

