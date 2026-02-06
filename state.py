from typing import Dict, Any

# In-memory state for jobs, uploaded file locations, modes, and analysis results.
jobs: Dict[str, str] = {}
# Uploaded file reference per job (file path string).
uploaded_files: Dict[str, str] = {}
# Optional mode per job, e.g. "normal" or "large" for large-file handling.
job_modes: Dict[str, str] = {}
# Per-job analysis metadata for export (started_at, completed_at, duration_ms, analysis_type, etc.).
analysis_meta: Dict[str, Dict[str, Any]] = {}
results_store: Dict[str, Dict[str, Any]] = {}

