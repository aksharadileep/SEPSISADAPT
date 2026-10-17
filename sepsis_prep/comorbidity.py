"""Static comorbidity flags (binary 0/1) for the treatment-recommendation state.

Source : hosp.diagnoses_icd  (ICD-9 and ICD-10 codes, stored WITHOUT dots, e.g. "40391", "I5022")
Link   : (subject_id, hadm_id)  ->  stay_id   (never subject_id alone)
Use    : constant context columns  s_<name>  (and ns_<name>) in the treatment table; NEVER actions.

How a code is matched
  * every rule is defined per ICD version (9 or 10) - a 9 code is never compared with a 10 rule
  * `prefixes`  : the code starts with one of these strings
  * `cat_range` : the 3-character category lies in [lo, hi]  (ICD-9: numeric 140..208 ; ICD-10: letter + 2 digits)
  * `exclude`   : prefixes removed again after the two rules above

Clinical scope (chronic conditions only; acute events are NOT comorbidities)
  diabetes               DM types 1/2/secondary (ICD-9 250, ICD-10 E08-E14).  Gestational DM and "prediabetes" excluded.
  heart_failure          HF + cardiomyopathy + hypertensive heart disease WITH HF (Quan/Elixhauser CHF list)
  chronic_kidney_disease CKD (any stage), ESRD, hypertensive CKD, dialysis dependence, kidney transplant status.
                         Acute kidney injury / unspecified renal failure (584, 586, N17, N19) are NOT included.
  chronic_liver_disease  cirrhosis, chronic hepatitis (incl. chronic viral B/C), portal hypertension, varices,
                         hepatorenal, chronic hepatic failure, liver transplant status.
                         Acute alcoholic hepatitis, acute hepatic failure and simple fatty liver are NOT included.
  cancer                 active-type malignancy: solid tumours, metastases, lymphoma, leukaemia, myeloma
                         (ICD-9 140-208 except 173 ; ICD-10 C00-C97 except C44, plus C7A/C7B).
                         Non-melanoma skin cancer and "history of cancer" (V10, Z85) are NOT included.
  immunosuppression      HIV/AIDS, primary/secondary immunodeficiency, transplant status / transplant complications
                         (solid organ, bone marrow, stem cell), GVHD, long-term systemic steroid or
                         immunosuppressant therapy. Neutropenia and cancer itself are NOT counted here.

Leakage note: diagnoses_icd is coded at hospital DISCHARGE and has no timestamp. The six conditions chosen are
chronic, and no sepsis / infection / acute-organ-failure code is used, but a few codes can still be first
documented during the admission. See README ("Comorbidities") for the honest limitation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("sepsis_prep")

COMORBIDITIES = [
    "diabetes",
    "heart_failure",
    "chronic_kidney_disease",
    "chronic_liver_disease",
    "cancer",
    "immunosuppression",
]
STATE_COMORBIDITY_COLS = [f"s_{c}" for c in COMORBIDITIES]
NEXT_COMORBIDITY_COLS = [f"ns_{c}" for c in COMORBIDITIES]

# rule = {icd_version: {"prefixes": (...), "cat_range": (lo, hi) | None, "exclude": (...)}}
RULES: dict[str, dict[int, dict]] = {
    "diabetes": {
        9: {"prefixes": ("250", "3572", "3620", "36641")},
        10: {"prefixes": ("E08", "E09", "E10", "E11", "E12", "E13", "E14")},
    },
    "heart_failure": {
        9: {"prefixes": ("428", "39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413",
                         "40491", "40493", "4254", "4255", "4256", "4257", "4258", "4259")},
        10: {"prefixes": ("I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427", "I428",
                          "I429", "I43", "I50")},
    },
    "chronic_kidney_disease": {
        9: {"prefixes": ("585", "403", "404", "V420", "V451", "V56")},
        10: {"prefixes": ("N18", "I12", "I13", "Z992", "Z940", "Z490", "Z491", "Z492")},
    },
    "chronic_liver_disease": {
        9: {"prefixes": ("5710", "5712", "5713", "5714", "5715", "5716", "5718", "5719",
                         "5722", "5723", "5724", "5728", "4560", "4561", "4562", "V427",
                         "07022", "07023", "07032", "07033", "07044", "07054")},
        10: {"prefixes": ("K70", "K713", "K714", "K715", "K717", "K721", "K73", "K74", "K754",
                          "K766", "K767", "K7681", "B18", "I850", "I851", "I864", "Z944"),
             "exclude": ("K701",)},                       # K70.1x = acute alcoholic hepatitis
    },
    "cancer": {
        9: {"prefixes": (), "cat_range": (140, 208), "exclude": ("173",)},
        10: {"prefixes": ("C4A", "C7A", "C7B"), "cat_range": (0, 97), "exclude": ("C44",)},
    },
    "immunosuppression": {
        9: {"prefixes": ("042", "V08", "2790", "2791", "2792", "2793", "2795",
                         "V420", "V421", "V426", "V427", "V4281", "V4282", "V4283", "V4284",
                         "9968", "V5865")},
        10: {"prefixes": ("B20", "Z21", "D80", "D81", "D82", "D83", "D84", "D8981",
                          "Z940", "Z941", "Z942", "Z943", "Z944", "Z9481", "Z9482", "Z9483", "Z9484",
                          "T86", "Z7952", "Z7960", "Z7962", "Z7969")},
    },
}


def _norm(code) -> str:
    return str(code).strip().upper().replace(".", "")


def _in_range(code: str, version: int, lo: int, hi: int) -> bool:
    if version == 9:
        head = code[:3]
        return head.isdigit() and lo <= int(head) <= hi
    # ICD-10 cancer chapter: letter 'C' + two digits
    return code[:1] == "C" and code[1:3].isdigit() and lo <= int(code[1:3]) <= hi


def code_matches(code, version, name: str) -> bool:
    """True if this single diagnosis code belongs to comorbidity `name`."""
    try:
        v = int(version)
    except (TypeError, ValueError):
        return False
    spec = RULES[name].get(v)
    if spec is None:
        return False
    c = _norm(code)
    if not c or c == "NAN":
        return False
    hit = c.startswith(spec["prefixes"]) if spec["prefixes"] else False
    rng = spec.get("cat_range")
    if not hit and rng is not None:
        hit = _in_range(c, v, rng[0], rng[1])
    if hit and spec.get("exclude") and c.startswith(tuple(spec["exclude"])):
        hit = False
    return hit


def flag_codes(dx: pd.DataFrame) -> pd.DataFrame:
    """dx(icd_version, icd_code, ...) -> same rows + 6 int8 flag columns (matching done once per unique code)."""
    uniq = dx[["icd_version", "icd_code"]].drop_duplicates().reset_index(drop=True)
    for name in COMORBIDITIES:
        uniq[name] = [int(code_matches(c, v, name)) for c, v in zip(uniq["icd_code"], uniq["icd_version"])]
    out = dx.merge(uniq, on=["icd_version", "icd_code"], how="left", validate="m:1")
    for name in COMORBIDITIES:
        out[name] = out[name].fillna(0).astype("int8")
    return out


def build_stay_comorbidities(dx: pd.DataFrame, cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pure function (unit-tested).

    dx     : subject_id, hadm_id, icd_version, icd_code   (rows of hosp.diagnoses_icd)
    cohort : subject_id, hadm_id, stay_id                 (one row per ICU stay)
    returns: (stay_id + 6 binary columns, one row per cohort stay ; report)

    The join key is ALWAYS (subject_id, hadm_id). A stay whose admission has no diagnosis rows gets 0 for all
    six flags ("not documented") - the count is reported so you can see how many stays that is.
    """
    keys = cohort[["stay_id", "subject_id", "hadm_id"]].copy()
    if keys["stay_id"].duplicated().any():
        raise ValueError("cohort has duplicate stay_id")
    for c in ("subject_id", "hadm_id", "stay_id"):
        keys[c] = keys[c].astype("int64")
    d = dx.copy()
    d["subject_id"], d["hadm_id"] = d["subject_id"].astype("int64"), d["hadm_id"].astype("int64")

    if keys.groupby("hadm_id")["subject_id"].nunique().max() > 1:
        raise ValueError("cohort: one hadm_id belongs to more than one subject_id")
    pair = keys[["subject_id", "hadm_id"]].drop_duplicates()
    d_ok = d.merge(pair, on=["subject_id", "hadm_id"], how="inner")              # BOTH keys must agree
    n_key_mismatch = int(len(d) - len(d_ok))
    # a hadm_id that belongs to another subject_id than the cohort says -> dropped here, counted in the report
    flagged = flag_codes(d_ok) if len(d_ok) else d_ok.assign(**{n: np.int8(0) for n in COMORBIDITIES})
    per_adm = flagged.groupby(["subject_id", "hadm_id"], as_index=False)[COMORBIDITIES].max()
    per_adm["_has_dx"] = 1

    out = keys.merge(per_adm, on=["subject_id", "hadm_id"], how="left", validate="m:1")
    no_dx = int(out["_has_dx"].isna().sum())
    for name in COMORBIDITIES:
        out[name] = out[name].fillna(0).astype("int8")
    out = out.rename(columns={n: f"s_{n}" for n in COMORBIDITIES})
    cols = ["stay_id"] + STATE_COMORBIDITY_COLS
    out = out[cols].sort_values("stay_id").reset_index(drop=True)

    assert len(out) == len(keys) and out["stay_id"].is_unique
    rep = {
        "n_stays": int(len(out)),
        "stays_without_any_diagnosis_row": no_dx,
        "diagnosis_rows_dropped_subject_hadm_mismatch": n_key_mismatch,
        "prevalence_pct": {c: round(100 * float(out[c].mean()), 3) for c in STATE_COMORBIDITY_COLS},
        "icd_rules": {k: {str(v): {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in s.items()}
                          for v, s in r.items()} for k, r in RULES.items()},
    }
    return out, rep


def validate_stay_table(tbl: pd.DataFrame) -> list[str]:
    """Binary, no NaN, one row per stay."""
    issues = []
    if tbl["stay_id"].duplicated().any():
        issues.append("comorbidity table: duplicate stay_id")
    for c in STATE_COMORBIDITY_COLS:
        if c not in tbl.columns:
            issues.append(f"comorbidity table: column {c} missing")
        elif tbl[c].isna().any() or not tbl[c].isin([0, 1]).all():
            issues.append(f"comorbidity table: {c} is not strictly 0/1")
    return issues
