"""Step 1: choose patients (sepsis + non-sepsis) and the time window of every stay."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import sql as S

log = logging.getLogger("sepsis_prep")


def _us(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).astype("datetime64[us]")


def choose_onset(df: pd.DataFrame, rule: str) -> pd.Series:
    inf, sofa = _us(df["suspected_infection_time"]), _us(df["sofa_time"])
    if rule == "latest":
        out = pd.concat([inf, sofa], axis=1).max(axis=1)
    elif rule == "earliest":
        out = pd.concat([inf, sofa], axis=1).min(axis=1)
    elif rule == "infection":
        out = inf
    elif rule == "sofa":
        out = sofa
    else:
        raise ValueError(f"onset_rule must be latest|earliest|infection|sofa, got {rule!r}")
    return out.astype("datetime64[us]")


def split_of(subject_id: int, seed: int, f_train: float, f_val: float) -> str:
    """Deterministic patient-level split (same patient -> same split, always)."""
    h = hashlib.md5(f"{seed}-{int(subject_id)}".encode()).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF
    return "train" if u < f_train else ("validation" if u < f_train + f_val else "test")


def assign_batches(is_sepsis, n_batches: int, seed: int = 42) -> np.ndarray:
    """Batch number 1..n_batches for every stay.

    Sepsis and non-sepsis stays are shuffled separately and dealt out round-robin, so EVERY batch
    contains both kinds of patients in (almost) the same proportion and all batches have the same size."""
    is_sepsis = np.asarray(is_sepsis, dtype=bool)
    rng = np.random.default_rng(seed)
    out = np.zeros(len(is_sepsis), dtype="int16")
    offset = 0
    for flag in (True, False):
        idx = np.flatnonzero(is_sepsis == flag)
        perm = rng.permutation(len(idx))
        out[idx[perm]] = (np.arange(len(idx)) + offset) % n_batches + 1
        offset += len(idx)
    return out


def refine_cohort(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Pure function (unit-tested): raw BigQuery rows -> final cohort + exclusion counts."""
    c, hc, sp = cfg["cohort"], cfg["hourly"], cfg["split"]
    df = raw.copy()
    for col in ("intime", "outtime"):
        df[col] = _us(df[col])
    rep = {"raw_stays_adult": int(len(df))}

    if c.get("first_icu_stay_only", True):
        df = df[df["stay_rank"] == 1]
    rep["after_first_stay_only"] = int(len(df))

    df = df[df["los_hours"] >= c["min_icu_hours"]]
    rep["after_min_icu_hours"] = int(len(df))

    df = df.copy()
    df["is_sepsis"] = df["is_sepsis"].fillna(False).astype(bool)
    df["sepsis_onset_time"] = pd.NaT
    m = df["is_sepsis"]
    df.loc[m, "sepsis_onset_time"] = choose_onset(df[m], c["onset_rule"]).values
    df["sepsis_onset_time"] = _us(df["sepsis_onset_time"])

    hrs_to_onset = (df["sepsis_onset_time"] - df["intime"]).dt.total_seconds() / 3600.0
    bad_missing = df["is_sepsis"] & df["sepsis_onset_time"].isna()
    bad_early = df["is_sepsis"] & (hrs_to_onset < c["min_hours_before_onset"])
    bad_late = df["is_sepsis"] & (df["sepsis_onset_time"] > df["outtime"])
    rep["sepsis_dropped_missing_onset"] = int(bad_missing.sum())
    rep["sepsis_dropped_onset_before_or_too_soon_after_icu_admission"] = int((bad_early & ~bad_missing).sum())
    rep["sepsis_dropped_onset_after_icu_discharge"] = int((bad_late & ~bad_missing & ~bad_early).sum())
    df = df[~(bad_missing | bad_early | bad_late)].copy()

    if c.get("control_ratio"):
        rng = np.random.default_rng(c.get("seed", 42))
        n_sep = int(df["is_sepsis"].sum())
        ctrl = df[~df["is_sepsis"]]
        keep = min(len(ctrl), int(n_sep * c["control_ratio"]))
        ctrl = ctrl.iloc[np.sort(rng.choice(len(ctrl), size=keep, replace=False))]
        df = pd.concat([df[df["is_sepsis"]], ctrl])
        rep["controls_subsampled_to"] = int(keep)

    if c.get("limit_stays"):
        n = int(c["limit_stays"])
        half = n // 2
        sep, ctl = df[df["is_sepsis"]].head(half), df[~df["is_sepsis"]].head(n - half)
        df = pd.concat([sep, ctl])

    # window of hourly rows for each stay
    df["window_start"] = df["intime"].dt.floor("h")
    cap_ctl = df["intime"] + pd.to_timedelta(hc["max_icu_hours_nonsepsis"], unit="h")
    end_ctl = pd.concat([df["outtime"], cap_ctl], axis=1).min(axis=1)
    end_sep = pd.concat([df["outtime"], df["sepsis_onset_time"] + pd.to_timedelta(hc["post_onset_hours"], unit="h")],
                        axis=1).min(axis=1)
    df["window_end"] = _us(end_sep.where(df["is_sepsis"], end_ctl))

    df["sex_male"] = (df["gender"] == "M").astype("int8")
    codes = {u: i for i, u in enumerate(sorted(df["first_careunit"].dropna().unique()))}
    df["careunit_code"] = df["first_careunit"].map(codes).fillna(-1).astype("int16")
    df["split"] = [split_of(s, sp["seed"], sp["train"], sp["validation"]) for s in df["subject_id"]]

    df = df.sort_values("stay_id").reset_index(drop=True)
    df["batch_id"] = assign_batches(df["is_sepsis"].values, int(hc["n_batches"]), c.get("seed", 42))
    keep_cols = ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "window_start", "window_end",
                 "is_sepsis", "sepsis_onset_time", "age", "sex_male", "careunit_code", "first_careunit",
                 "hospital_expire_flag", "split", "batch_id"]
    out = df[keep_cols].copy()
    out["age"] = out["age"].astype("float32")
    rep["final_stays"] = int(len(out))
    rep["final_sepsis_stays"] = int(out["is_sepsis"].sum())
    rep["final_nonsepsis_stays"] = int((~out["is_sepsis"]).sum())
    per_batch = out.groupby("batch_id")["is_sepsis"].agg(["size", "sum"])
    rep["n_batches"] = int(len(per_batch))
    rep["stays_per_batch_min_max"] = [int(per_batch["size"].min()), int(per_batch["size"].max())]
    rep["sepsis_stays_per_batch_min_max"] = [int(per_batch["sum"].min()), int(per_batch["sum"].max())]
    rep["careunit_codes"] = codes
    rep["stays_per_split"] = {k: int(v) for k, v in out["split"].value_counts().items()}
    rep["sepsis_per_split"] = {k: int(v) for k, v in out[out["is_sepsis"]]["split"].value_counts().items()}
    return out, rep


def build_cohort(bq, cfg: dict, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Step 1/3  cohort: querying BigQuery (icustays + patients + admissions + sepsis3) ...")
    raw = bq.query_df(S.cohort_sql(bq), {"min_age": int(cfg["cohort"]["min_age"])}, label="cohort")
    cohort, rep = refine_cohort(raw, cfg)
    if cohort.empty:
        raise RuntimeError("Cohort is empty - check bigquery dataset names in config.yaml")
    cohort.to_parquet(out_dir / "cohort.parquet", index=False)
    h = hashlib.md5(pd.util.hash_pandas_object(cohort[["stay_id", "window_start", "window_end", "batch_id"]], index=False).values.tobytes()).hexdigest()
    rep["cohort_hash"] = h
    (out_dir / "cohort_report.json").write_text(json.dumps(rep, indent=2))
    log.info("cohort: %d stays  (%d sepsis / %d non-sepsis)", rep["final_stays"], rep["final_sepsis_stays"], rep["final_nonsepsis_stays"])
    return cohort
