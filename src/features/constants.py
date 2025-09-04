from pathlib import Path
import os

# Defaults (kept minimal and safe)
DEFAULT_DB_PATH = Path(os.getenv("IV_DB_PATH", "data/iv_data_1h.db"))  # preserve your original default
ANNUAL_HOURS = 252 * 6.5
ANNUAL_MINUTES = 252 * 390

# Only the essential hide rules that matter for leakage
HIDE_COLUMNS = {
    "iv_ret_fwd": ["iv_ret_fwd_abs", "core_iv_ret_fwd_abs"],
    "iv_ret_fwd_abs": ["iv_ret_fwd"],
    "iv_clip": ["iv_ret_fwd", "iv_ret_fwd_abs", "core_iv_ret_fwd_abs"],
}

# Optional: keep here for reference if you need in UI or checks
CORE_FEATURE_COLS = []
