"""
Clio Core Drive Failure Prediction API

Serves pre-trained LightGBM models for HDD (Backblaze 2023 data) and SSD
(Alibaba dcbrain data) failure prediction.

Endpoints:
  POST /predict/auto    - Automatically chooses static or rolling based on drive history
  POST /predict/static  - Single-snapshot failure prediction + estimated failure date
  POST /predict/rolling - Time-windowed (delta-feature) prediction (requires 14 days history)
  GET  /health          - Liveness check + model status
  GET  /drive/{id}/history - View accumulated history for a specific drive

Architecture:
  - Every call to /predict/auto stores the incoming SMART metrics in a local SQLite
    database keyed by drive_id.
  - Once a drive has >= 14 days of readings, delta features are computed automatically
    and the rolling model is used instead of the static model.
  - History persists across server restarts so drives are never reset.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import sqlite3
import math
import os
import json
import uvicorn
import threading
from datetime import date, timedelta, datetime

app = FastAPI(
    title="Clio Core Drive Failure Prediction API",
    description="Pre-trained drive failure prediction for HDDs and SSDs with automatic rolling model promotion",
    version="2.0.0"
)

# ---------------------------------------------------------------------------
# Weibull-calibrated survival parameters
# (shape k, scale lambda in days) fitted to Backblaze annual survival curves.
# HDD: mean life ~4.2 years, shape k=2.1  → lambda ≈ 1730 days
# SSD: mean life ~5.8 years, shape k=1.8  → lambda ≈ 2400 days
# ---------------------------------------------------------------------------
WEIBULL_PARAMS = {
    'hdd': {'k': 2.1, 'lam': 1730.0},
    'ssd': {'k': 1.8, 'lam': 2400.0},
}

# Confidence band width (±days around the point estimate)
CONFIDENCE_BAND = {
    'hdd': {'low': 0.25, 'high': 0.35},   # ± ~30 %
    'ssd': {'low': 0.30, 'high': 0.40},   # ± ~35 % (synthetic model, wider)
}

# Minimum probability threshold below which we say "no imminent failure"
P_MIN_ACTIONABLE = 0.02

# Number of days of live history required before promoting a drive to the rolling model.
# Must match the rolling_days window used during training.
ROLLING_MIN_DAYS = 7

# SMART feature names (canonical — without _raw suffix), must match train.py
SMART_FEATURES = {
    'hdd': [
        'smart_5', 'smart_187', 'smart_188', 'smart_197', 'smart_198',
        'smart_1', 'smart_7', 'smart_9', 'smart_10', 'smart_192', 'smart_193', 'smart_194',
    ],
    'ssd': [
        'smart_5', 'smart_9', 'smart_173', 'smart_174', 'smart_177',
        'smart_183', 'smart_187', 'smart_194', 'smart_230', 'smart_241', 'smart_242',
    ],
}

# Path to the SQLite drive history database
DB_PATH = os.environ.get('DRIVE_HISTORY_DB', 'drive_history.db')

# Thread lock for SQLite writes
_db_lock = threading.Lock()

# Global model registry
models = {
    'hdd': {'static': None, 'rolling': None},
    'ssd': {'static': None, 'rolling': None},
}


class PredictionRequest(BaseModel):
    """
    Prediction request payload.

    Args:
        drive_type: 'hdd' or 'ssd'.
        metrics:    Dict of SMART attribute key-value pairs.
        drive_id:   Optional unique drive identifier. Required for /predict/auto
                    to enable per-drive history accumulation and rolling model
                    promotion. If omitted, falls back to static model only.
    """
    drive_type: str
    metrics: dict
    drive_id: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Normalize _raw suffix: the C++ health monitor writes keys like
        # "smart_5_raw" but the model was trained on canonical "smart_5".
        # Strip the suffix so both formats are accepted transparently.
        if self.metrics:
            self.metrics = {
                (k[:-4] if k.endswith('_raw') else k): v
                for k, v in self.metrics.items()
            }


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Initialize the SQLite database schema for per-drive SMART history.

    Creates a 'drive_readings' table if it does not already exist.
    Each row represents one reading (one day) for one drive.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS drive_readings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                drive_id    TEXT    NOT NULL,
                drive_type  TEXT    NOT NULL,
                reading_ts  TEXT    NOT NULL,
                metrics_json TEXT   NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_drive_id ON drive_readings(drive_id, reading_ts)
        """)
        conn.commit()


def store_reading(drive_id: str, drive_type: str, metrics: dict) -> None:
    """
    Persist one SMART reading for a drive in the SQLite history database.

    Args:
        drive_id:   Unique identifier for the drive (e.g. serial number).
        drive_type: 'hdd' or 'ssd'.
        metrics:    Dict of SMART attribute values from this reading.
    """
    import json
    ts = datetime.utcnow().isoformat()
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO drive_readings (drive_id, drive_type, reading_ts, metrics_json) VALUES (?,?,?,?)",
                (drive_id, drive_type, ts, json.dumps(metrics))
            )
            conn.commit()


def get_drive_history(drive_id: str) -> pd.DataFrame:
    """
    Retrieve the full chronological SMART reading history for a drive.

    Args:
        drive_id: Unique drive identifier.

    Returns:
        DataFrame sorted by reading timestamp with one column per SMART attribute.
        Returns empty DataFrame if the drive has no history.
    """
    import json
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT reading_ts, metrics_json FROM drive_readings WHERE drive_id=? ORDER BY reading_ts ASC",
            (drive_id,)
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    records = []
    for ts, metrics_json in rows:
        row = json.loads(metrics_json)
        row['reading_ts'] = ts
        records.append(row)
    return pd.DataFrame(records)


def count_drive_readings(drive_id: str) -> int:
    """
    Return how many readings have been stored for a given drive.

    Args:
        drive_id: Unique drive identifier.

    Returns:
        Number of stored readings (days of history).
    """
    with sqlite3.connect(DB_PATH) as conn:
        result = conn.execute(
            "SELECT COUNT(*) FROM drive_readings WHERE drive_id=?", (drive_id,)
        ).fetchone()
    return result[0] if result else 0


def load_models(model_dir: str = 'models') -> None:
    """
    Load pre-trained .pkl models and fitted Weibull params from disk.

    Weibull params are loaded from {drive_type}_weibull_params.json if present,
    overriding the hardcoded defaults. This ensures the failure date estimates
    are calibrated to the actual training data rather than population averages.

    Args:
        model_dir: Directory containing model pkl files and weibull json files.
    """
    for dt in ('hdd', 'ssd'):
        for kind in ('static', 'rolling'):
            path = os.path.join(model_dir, f'{dt}_{kind}_model.pkl')
            if os.path.exists(path):
                models[dt][kind] = joblib.load(path)
                print(f"Loaded {dt} {kind} model from {path}")
            else:
                print(f"Warning: model not found at {path}")

        # Load fitted Weibull params if available (Fix #3)
        weibull_path = os.path.join(model_dir, f'{dt}_weibull_params.json')
        if os.path.exists(weibull_path):
            with open(weibull_path) as f:
                fitted = json.load(f)
            WEIBULL_PARAMS[dt] = fitted
            print(f"Loaded fitted Weibull params for {dt}: {fitted}")
        else:
            print(f"Using default Weibull params for {dt}: {WEIBULL_PARAMS[dt]}")


def estimate_failure_date(failure_prob: float, drive_type: str) -> dict:
    """
    Estimate days until failure using an inverse-Weibull (quantile) approach.

    Given a failure probability p, we invert the Weibull CDF to get the
    corresponding time t:

        F(t) = 1 - exp(-(t/lambda)^k)  =>  t = lambda * (-ln(1-p))^(1/k)

    We treat the model output as the cumulative failure probability already
    accumulated by this drive over its life, so t represents residual life
    from *today*.

    Args:
        failure_prob: Probability of imminent failure (0.0–1.0) from the model.
        drive_type:   'hdd' or 'ssd'.

    Returns:
        Dict with 'days_remaining', 'estimated_failure_date',
        'confidence_lower', 'confidence_upper', and 'risk_level'.
    """
    today = date.today()

    if failure_prob < P_MIN_ACTIONABLE:
        return {
            "days_remaining": None,
            "estimated_failure_date": None,
            "confidence_lower": None,
            "confidence_upper": None,
            "risk_level": "healthy",
            "risk_description": "No imminent failure detected. Drive appears healthy."
        }

    # Clamp to avoid log(0) or log(negative)
    p = min(max(failure_prob, 1e-6), 1.0 - 1e-9)

    params = WEIBULL_PARAMS.get(drive_type, WEIBULL_PARAMS['hdd'])
    k = params['k']
    lam = params['lam']

    # Inverse Weibull (point estimate)
    days = lam * ((-math.log(1.0 - p)) ** (1.0 / k))
    days = max(1.0, days)

    # Confidence band
    band = CONFIDENCE_BAND.get(drive_type, CONFIDENCE_BAND['hdd'])
    lower_days = max(1.0, days * (1.0 - band['low']))
    upper_days = days * (1.0 + band['high'])

    # Risk classification
    if days <= 14:
        risk_level = "critical"
        risk_desc = f"Drive failure expected imminently (~{int(days)} days). Replace immediately."
    elif days <= 60:
        risk_level = "high"
        risk_desc = f"High failure risk. Plan replacement within {int(days)} days."
    elif days <= 180:
        risk_level = "moderate"
        risk_desc = f"Moderate failure risk. Monitor closely, plan replacement within {int(days)} days."
    else:
        risk_level = "low"
        risk_desc = f"Low but non-zero failure risk. Next checkpoint: {int(days)} days."

    return {
        "days_remaining": int(days),
        "estimated_failure_date": str(today + timedelta(days=int(days))),
        "confidence_lower": str(today + timedelta(days=int(lower_days))),
        "confidence_upper": str(today + timedelta(days=int(upper_days))),
        "risk_level": risk_level,
        "risk_description": risk_desc,
    }


def run_static_prediction(model, req: PredictionRequest) -> dict:
    """
    Run static snapshot inference using only the current SMART reading.

    Args:
        model: Loaded LightGBM static classifier.
        req:   Parsed PredictionRequest with current metrics.

    Returns:
        Dict containing failure_probability and all failure date fields.
    """
    df = pd.DataFrame([req.metrics])
    for feat in model.feature_name_:
        if feat not in df.columns:
            df[feat] = 0.0
    df = df[model.feature_name_]
    probs = model.predict_proba(df)[0]
    failure_prob = float(probs[1]) if len(probs) > 1 else 0.0
    date_info = estimate_failure_date(failure_prob, req.drive_type.lower())
    return {"drive_type": req.drive_type.lower(), "failure_probability": round(failure_prob, 4), **date_info}


def run_rolling_prediction(model, history_df: pd.DataFrame, drive_type: str) -> dict:
    """
    Run rolling-window inference using 14-day delta features from stored history.

    Args:
        model:      Loaded LightGBM rolling classifier.
        history_df: DataFrame of chronological SMART readings for this drive.
        drive_type: 'hdd' or 'ssd'.

    Returns:
        Dict containing failure_probability and all failure date fields.
    """
    features = SMART_FEATURES.get(drive_type, SMART_FEATURES['hdd'])
    for f in features:
        if f not in history_df.columns:
            history_df[f] = 0.0
    tail = history_df[features].tail(ROLLING_MIN_DAYS + 1).reset_index(drop=True)
    latest = tail.iloc[-1]
    oldest = tail.iloc[0]
    delta_row = {}
    for f in features:
        delta_row[f] = float(latest[f])
        delta_row[f'{f}_delta'] = float(latest[f]) - float(oldest[f])
    df = pd.DataFrame([delta_row])
    for feat in model.feature_name_:
        if feat not in df.columns:
            df[feat] = 0.0
    df = df[model.feature_name_]
    probs = model.predict_proba(df)[0]
    failure_prob = float(probs[1]) if len(probs) > 1 else 0.0
    date_info = estimate_failure_date(failure_prob, drive_type)
    return {"drive_type": drive_type, "failure_probability": round(failure_prob, 4), **date_info}


@app.on_event("startup")
def startup():
    """Initialize DB and load models on server start."""
    init_db()
    model_dir = os.environ.get('MODEL_DIR', 'models')
    load_models(model_dir)


@app.get("/health")
def health():
    """Liveness check — returns which models are loaded and DB path."""
    return {
        "status": "ok",
        "db_path": DB_PATH,
        "models_loaded": {
            dt: {kind: m is not None for kind, m in kinds.items()}
            for dt, kinds in models.items()
        }
    }


@app.post("/predict/auto")
def predict_auto(req: PredictionRequest):
    """
    Automatically chooses static or rolling model based on accumulated drive history.

    - If drive_id is provided, stores the current reading in SQLite.
    - If the drive has >= 14 days of history AND a rolling model is loaded,
      promotes to rolling model automatically.
    - Otherwise uses the static snapshot model.
    - Returns which model was used in the 'model_used' field.
    """
    dt = req.drive_type.lower()
    if dt not in models:
        raise HTTPException(status_code=400, detail="Invalid drive_type. Use 'hdd' or 'ssd'.")
    static_model = models[dt]['static']
    rolling_model = models[dt]['rolling']
    if static_model is None:
        raise HTTPException(status_code=503, detail=f"Static model for '{dt}' is not loaded.")

    # Store this reading if we have a drive_id
    if req.drive_id:
        store_reading(req.drive_id, dt, req.metrics)
        num_readings = count_drive_readings(req.drive_id)
    else:
        num_readings = 0

    # Promote to rolling model if we have enough history
    if rolling_model is not None and req.drive_id and num_readings >= ROLLING_MIN_DAYS:
        history_df = get_drive_history(req.drive_id)
        try:
            result = run_rolling_prediction(rolling_model, history_df, dt)
            result['model_used'] = 'rolling'
            result['days_of_history'] = num_readings
            return result
        except Exception as e:
            print(f"Rolling model error for {req.drive_id}, falling back: {e}")

    # Static fallback
    try:
        result = run_static_prediction(static_model, req)
        result['model_used'] = 'static'
        result['days_of_history'] = num_readings
        result['days_until_rolling'] = max(0, ROLLING_MIN_DAYS - num_readings) if req.drive_id else None
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/static")
def predict_static(req: PredictionRequest):
    """
    Single-snapshot failure prediction (always uses static model, no history stored).
    """
    dt = req.drive_type.lower()
    if dt not in models:
        raise HTTPException(status_code=400, detail="Invalid drive_type. Use 'hdd' or 'ssd'.")
    model = models[dt]['static']
    if model is None:
        raise HTTPException(status_code=503, detail=f"Static model for '{dt}' is not loaded.")
    try:
        result = run_static_prediction(model, req)
        result['model_used'] = 'static'
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/rolling")
def predict_rolling(req: PredictionRequest):
    """
    Rolling-window prediction. Requires drive_id with at least 14 days of stored history.
    """
    dt = req.drive_type.lower()
    if dt not in models:
        raise HTTPException(status_code=400, detail="Invalid drive_type. Use 'hdd' or 'ssd'.")
    model = models[dt]['rolling']
    if model is None:
        raise HTTPException(status_code=503, detail=f"Rolling model for '{dt}' is not loaded.")
    if not req.drive_id:
        raise HTTPException(status_code=400, detail="drive_id is required for rolling prediction.")
    num_readings = count_drive_readings(req.drive_id)
    if num_readings < ROLLING_MIN_DAYS:
        raise HTTPException(
            status_code=422,
            detail=f"Drive '{req.drive_id}' only has {num_readings} days of history. "
                   f"Need {ROLLING_MIN_DAYS}. Use /predict/auto to accumulate history."
        )
    history_df = get_drive_history(req.drive_id)
    try:
        result = run_rolling_prediction(model, history_df, dt)
        result['model_used'] = 'rolling'
        result['days_of_history'] = num_readings
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drive/{drive_id}/history")
def drive_history(drive_id: str):
    """
    Return the stored reading count and timestamps for a drive.
    Shows how close a drive is to rolling model promotion.
    """
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT COUNT(*), MIN(reading_ts), MAX(reading_ts) FROM drive_readings WHERE drive_id=?",
            (drive_id,)
        ).fetchone()
    count, first_ts, last_ts = row if row else (0, None, None)
    return {
        "drive_id": drive_id,
        "total_readings": count,
        "first_reading": first_ts,
        "last_reading": last_ts,
        "rolling_model_eligible": count >= ROLLING_MIN_DAYS,
        "days_until_rolling": max(0, ROLLING_MIN_DAYS - count),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--db', type=str, default='drive_history.db')
    args = parser.parse_args()
    os.environ['MODEL_DIR'] = args.model_dir
    os.environ['DRIVE_HISTORY_DB'] = args.db
    DB_PATH = args.db
    init_db()
    load_models(args.model_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
