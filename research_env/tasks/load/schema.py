"""Constants, tracks, request construction and prediction validation.

A *record* is one person-condition row produced by the trusted importer. It
carries private fields (reference power, laboratory variables, source person
ID). A *request* is the only thing a participant artifact ever sees at
inference time; it is built here from an explicit allow-list, never by
deleting fields from a record.
"""
from __future__ import annotations

import math

TASK_ID = "load-evobreath-v1"

# Unit conversions. 1 kcal = 4184 J (thermochemical calorie); 1 day = 86400 s.
J_PER_KCAL = 4184.0
S_PER_DAY = 86400.0
SPEED_KM_H = 4.0
SPEED_M_S = SPEED_KM_H / 3.6
GRADE_FRACTION = 0.0
LOADS_KG = (0.0, 5.0, 10.0, 15.0)
LOAD_COLUMNS = {0.0: "EE_WW00", 5.0: "EE_WW05", 10.0: "EE_WW10", 15.0: "EE_WW15"}


def kcal_per_day_to_watts(kcal_per_day: float) -> float:
    return kcal_per_day * J_PER_KCAL / S_PER_DAY


# Inputs a deployed planner could plausibly know without a laboratory visit.
PRIMARY_FEATURES = ("body_mass_kg", "stature_cm", "age_years", "load_kg")
OPTIONAL_COVARIATES = ("sex",)
FIXED_CONTEXT = ("speed_m_s", "grade_fraction")
# Laboratory measurements never sent to primary-tier artifacts.
LAB_ONLY = ("femur_length_cm", "biiliac_breadth_cm", "fat_mass_kg",
            "fat_free_mass_kg", "rmr_kcal_day", "rmr_w")

TRACKS = {
    # New people, all recorded loads, no measured metabolic input.
    "primary": {"train_loads": LOADS_KG, "score_loads": LOADS_KG, "calibration": False},
    # New people; the 0 kg measurement is revealed; loaded conditions are scored.
    "calibration": {"train_loads": LOADS_KG, "score_loads": (5.0, 10.0, 15.0), "calibration": True},
    # New people and a load never seen in training.
    "holdout15": {"train_loads": (0.0, 5.0, 10.0), "score_loads": (15.0,), "calibration": False},
}

# Sanity bound only; wide-but-finite intervals are penalized by IS90, not rejected.
MAX_ABS_WATTS = 1.0e6


def build_request(record: dict, track: str, person_records: list[dict] | None = None,
                  request_id: str | None = None) -> dict:
    """Allow-list request for one record. person_records is needed only for calibration."""
    cfg = TRACKS[track]
    request = {
        "request_id": request_id,
        "track": track,
        "body_mass_kg": record["body_mass_kg"],
        "stature_cm": record["stature_cm"],
        "age_years": record["age_years"],
        "sex": record["sex"],
        "load_kg": record["load_kg"],
        "speed_m_s": record["speed_m_s"],
        "grade_fraction": record["grade_fraction"],
        "calibration": None,
    }
    if cfg["calibration"]:
        unloaded = [r for r in (person_records or [])
                    if r["load_kg"] == 0.0 and r["reference_w"] is not None]
        if len(unloaded) != 1:
            raise ValueError("calibration request needs exactly one measured 0 kg record")
        request["calibration"] = {"load_kg": 0.0, "measured_w": unloaded[0]["reference_w"]}
    return request


def validate_prediction(pred) -> str | None:
    """Return a failure reason, or None if the prediction is a valid 90% interval forecast."""
    if not isinstance(pred, dict):
        return "prediction is not an object"
    try:
        lo, mu, hi = (float(pred[k]) for k in ("lower_w", "mean_w", "upper_w"))
    except (KeyError, TypeError, ValueError):
        return "prediction needs numeric mean_w, lower_w, upper_w"
    if not all(math.isfinite(v) for v in (lo, mu, hi)):
        return "non-finite prediction"
    if not lo <= mu <= hi:
        return "requires lower_w <= mean_w <= upper_w"
    if max(abs(lo), abs(hi)) > MAX_ABS_WATTS:
        return f"prediction magnitude exceeds {MAX_ABS_WATTS} W"
    return None
