"""Single place that lists every clinical variable and where it comes from."""
from __future__ import annotations

VITALS = ["heart_rate", "sbp", "dbp", "mbp", "resp_rate", "temperature", "spo2"]

# derived table name -> columns
LAB_TABLES: dict[str, list[str]] = {
    "bg": ["lactate", "ph", "po2", "pco2", "baseexcess", "pao2fio2ratio"],
    "chemistry": ["creatinine", "bun", "sodium", "potassium", "chloride", "bicarbonate",
                  "glucose", "aniongap", "calcium", "albumin"],
    "complete_blood_count": ["wbc", "platelet", "hemoglobin", "hematocrit"],
    "enzyme": ["bilirubin_total", "alt", "ast"],
    "coagulation": ["inr", "ptt"],
}

# feature -> (source key, family)   family decides the forward-fill limit
FEATURES: dict[str, tuple[str, str]] = {}
for _v in VITALS:
    FEATURES[_v] = ("vitals", "vitals")
FEATURES["gcs"] = ("gcs", "vitals")
for _t, _cols in LAB_TABLES.items():
    for _c in _cols:
        FEATURES[_c] = (_t, "labs")

CLINICAL = list(FEATURES)                      # measured values (can be missing)
TREATMENT_HOURLY = ["fluid_ml", "vaso_ne_equiv", "vaso_on", "mech_vent", "urine_ml"]
STATIC = ["age", "sex_male", "weight_kg", "careunit_code"]
OPTIONAL = ["sofa_24h"]                        # derived from the same data; see README

ID_COLS = ["subject_id", "hadm_id", "stay_id", "timestamp", "hour", "hours_since_icu_admission", "split"]

# Columns that need a median when still missing after carry-forward
MEDIAN_COLS = CLINICAL + ["weight_kg", "sofa_24h"]


def label_cols(horizons) -> list[str]:
    return [f"label_sepsis_{h}h" for h in horizons]
