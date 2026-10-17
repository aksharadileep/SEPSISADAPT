"""Step 3: raw batches -> model-ready files (train-only imputation, patient-level splits, validation)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .comorbidity import NEXT_COMORBIDITY_COLS, STATE_COMORBIDITY_COLS, validate_stay_table
from .features import CLINICAL, MEDIAN_COLS, OPTIONAL, STATIC, TREATMENT_HOURLY, label_cols
from . import treatment as T

log = logging.getLogger("sepsis_prep")
SPLITS = ["train", "validation", "test"]
META = ["subject_id", "hadm_id", "stay_id", "batch_id", "timestamp", "hour", "hours_since_icu_admission", "split",
        "is_sepsis_stay", "hours_to_onset"]


def feature_lists(cfg: dict) -> dict:
    flags = [f"{c}_obs" for c in CLINICAL] if cfg["hourly"].get("add_obs_flags", True) else []
    model_inputs = CLINICAL + flags + TREATMENT_HOURLY + ["urine_obs"] + STATIC
    return {
        "model_inputs": model_inputs,
        "optional_inputs": OPTIONAL,
        "labels": label_cols(cfg["hourly"]["horizons"]),
        "meta_never_use_as_input": META,
        "treatment_state_comorbidity": STATE_COMORBIDITY_COLS,       # RL context features (static, 0/1), never actions
        "treatment_next_state_comorbidity": NEXT_COMORBIDITY_COLS if cfg.get("comorbidities", {}).get("add_next_state_columns", True) else [],
    }


def collect_stats(batches: list[Path], cfg: dict, per_batch_cap: int = 20000) -> dict:
    """Medians / mean / std from TRAIN rows only (real measurements only) + action bin cutpoints."""
    rng = np.random.default_rng(0)
    vals: dict[str, list] = {c: [] for c in MEDIAN_COLS}
    blocks = []
    for i, b in enumerate(batches, 1):
        df = pd.read_parquet(b)
        tr = df[df["split"] == "train"]
        for c in MEDIAN_COLS:
            if c in CLINICAL:
                v = tr.loc[tr[f"{c}_obs"] == 1, c].dropna().values
            else:
                v = tr[c].dropna().values
            if len(v) > per_batch_cap:
                v = rng.choice(v, per_batch_cap, replace=False)
            vals[c].append(v.astype("float64"))
        if tr["is_sepsis"].any():
            blocks.append(T.block_actions(tr, cfg))
        if i % 10 == 0:
            log.info("stats: %d/%d batches", i, len(batches))
    stats = {"median": {}, "mean": {}, "std": {}, "missing_in_train": []}
    for c, parts in vals.items():
        v = np.concatenate(parts) if parts else np.array([])
        if len(v) == 0:
            stats["missing_in_train"].append(c)
            stats["median"][c], stats["mean"][c], stats["std"][c] = 0.0, 0.0, 1.0
        else:
            stats["median"][c], stats["mean"][c] = float(np.median(v)), float(v.mean())
            stats["std"][c] = float(v.std() or 1.0)
    stats["action_cutpoints"] = T.action_cutpoints(pd.concat(blocks) if blocks else pd.DataFrame({"fluid_block": [], "vaso_block": []}),
                                                   int(cfg["treatment"]["n_bins"]))
    if stats["missing_in_train"]:
        log.warning("NEVER measured in train (filled with 0): %s", stats["missing_in_train"])
    return stats


def impute(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    for c, m in stats["median"].items():
        if c in df.columns:
            df[c] = df[c].fillna(np.float32(m))
    return df


def prediction_frame(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fl = feature_lists(cfg)
    p = df[df["already_septic"] == 0].rename(columns={"is_sepsis": "is_sepsis_stay"})
    cols = META + fl["model_inputs"] + fl["optional_inputs"] + fl["labels"]
    return p[cols].reset_index(drop=True)


def validate_part(p: pd.DataFrame, cfg: dict, subj_split: dict, issues: list) -> None:
    fl = feature_lists(cfg)
    if p.duplicated(["stay_id", "hour"]).any():
        issues.append("duplicate (stay_id, hour) rows")
    if p[fl["model_inputs"] + fl["optional_inputs"]].isna().any().any():
        issues.append("NaN left in features after imputation")
    for h in cfg["hourly"]["horizons"]:
        c = f"label_sepsis_{h}h"
        pos = p[p[c] == 1]
        if len(pos) and (~pos["is_sepsis_stay"].all() or not ((pos["hours_to_onset"] > 0) & (pos["hours_to_onset"] <= h)).all()):
            issues.append(f"{c}: label is 1 outside (0, {h}h] before onset")
    if (p.loc[~p["is_sepsis_stay"], fl["labels"]].values != 0).any():
        issues.append("non-sepsis stay has a positive label")
    for s, sp in p.groupby("subject_id")["split"].first().items():
        if subj_split.setdefault(int(s), sp) != sp:
            issues.append(f"patient {s} appears in 2 splits")


def validate_comorbidity_steps(steps: pd.DataFrame, cfg: dict, issues: list) -> None:
    """Every treatment row: 6 flags present, strictly 0/1, no NaN, constant within each stay (state AND next state)."""
    if steps.empty:
        return
    cols = list(STATE_COMORBIDITY_COLS)
    if cfg.get("comorbidities", {}).get("add_next_state_columns", True):
        cols += NEXT_COMORBIDITY_COLS
    miss = [c for c in cols if c not in steps.columns]
    if miss:
        issues.append(f"treatment table lacks comorbidity columns {miss}")
        return
    if steps[cols].isna().any().any() or not steps[cols].isin([0, 1]).all().all():
        issues.append("comorbidity columns not strictly 0/1 / contain NaN")
    if (steps.groupby("stay_id")[cols].nunique() > 1).any().any():
        issues.append("a stay has changing comorbidity values across its treatment rows")
    if cfg.get("comorbidities", {}).get("add_next_state_columns", True):
        if any((steps[s_] != steps[n_]).any() for s_, n_ in zip(STATE_COMORBIDITY_COLS, NEXT_COMORBIDITY_COLS)):
            issues.append("ns_<comorbidity> differs from s_<comorbidity>")
    if steps.duplicated(["stay_id", "step"]).any():
        issues.append("duplicate (stay_id, step) treatment rows")


def write_table(df: pd.DataFrame, base: Path, fmt: str) -> None:
    if fmt == "csv":
        df.to_csv(base.with_suffix(".csv"), index=False)
    else:
        df.to_parquet(base.with_suffix(".parquet"), index=False)


def finalize(cfg: dict, out_dir: Path) -> dict:
    """raw hourly batches -> output/early_prediction/batch_XX + output/treatment_recommendation/batch_XX"""
    pc = cfg["paths"]
    fmt = pc.get("file_format", "parquet")
    raw_dir = out_dir / pc.get("work_dir", "_work") / "raw_batches"
    batches = sorted(raw_dir.glob("batch_*.parquet"))
    if not batches:
        raise RuntimeError("No batches found - run `python run.py extract` first")
    early, treat = out_dir / pc["early_dir"], out_dir / pc["treatment_dir"]
    for d in (early, treat):
        d.mkdir(parents=True, exist_ok=True)
        for old in list(d.glob("batch_*.parquet")) + list(d.glob("batch_*.csv")):
            old.unlink()                                   # only our own batch files, nothing else

    comorb = None
    if cfg.get("comorbidities", {}).get("enabled", False):
        cp = out_dir / pc.get("work_dir", "_work") / "comorbidities.parquet"
        if not cp.exists():
            raise RuntimeError(f"{cp} not found. Run: python run.py comorbidities   (or set comorbidities.enabled: false)")
        comorb = pd.read_parquet(cp)
        bad = validate_stay_table(comorb)
        if bad:
            raise RuntimeError("comorbidity table invalid: " + "; ".join(bad))

    log.info("Step 3/3  finalize: pass 1 - train statistics from %d batches", len(batches))
    stats = collect_stats(batches, cfg)
    (out_dir / "train_stats.json").write_text(json.dumps(stats, indent=2))
    (out_dir / "feature_columns.json").write_text(json.dumps(feature_lists(cfg), indent=2))

    log.info("pass 2 - impute, build the two datasets, write one file per batch")
    horizons = cfg["hourly"]["horizons"]
    rep = {s: {"stays": 0, "sepsis_stays": 0, "pred_rows": 0, **{f"pos_rows_{h}h": 0 for h in horizons}, "treatment_steps": 0} for s in SPLITS}
    per_batch: list[dict] = []
    issues: list[str] = []
    subj_split: dict[int, str] = {}
    steps_seen: list[pd.DataFrame] = []
    for i, b in enumerate(batches, 1):
        k = int(b.stem.split("_")[1])                      # batch number (1..n_batches)
        df = impute(pd.read_parquet(b), stats)
        pred = prediction_frame(df, cfg)
        validate_part(pred, cfg, subj_split, issues)
        steps = T.build_steps(df, cfg, stats["action_cutpoints"], comorb)
        if comorb is not None:
            validate_comorbidity_steps(steps, cfg, issues)
        if len(steps):
            steps_seen.append(steps[["stay_id", "split"] + (STATE_COMORBIDITY_COLS if comorb is not None else [])].drop_duplicates("stay_id"))

        n_sep = int(pred.loc[pred["is_sepsis_stay"], "stay_id"].nunique())
        n_non = int(pred.loc[~pred["is_sepsis_stay"], "stay_id"].nunique())
        if n_sep == 0 or n_non == 0:
            issues.append(f"batch {k} has only one kind of patient (sepsis={n_sep}, non-sepsis={n_non})")
        write_table(pred, early / f"batch_{k:02d}", fmt)
        if len(steps):
            write_table(steps, treat / f"batch_{k:02d}", fmt)
        else:
            log.warning("batch %d: no treatment steps (no sepsis patient had a usable window)", k)

        per_batch.append({"batch": k, "stays": n_sep + n_non, "sepsis_stays": n_sep, "non_sepsis_stays": n_non,
                          "prediction_rows": int(len(pred)), "positive_rows_6h": int(pred["label_sepsis_6h"].sum()) if 6 in horizons else None,
                          "treatment_steps": int(len(steps)), "treatment_stays": int(steps["stay_id"].nunique()) if len(steps) else 0})
        for s in SPLITS:
            ps = pred[pred["split"] == s]
            r = rep[s]
            r["pred_rows"] += len(ps)
            r["stays"] += ps["stay_id"].nunique()
            r["sepsis_stays"] += ps.loc[ps["is_sepsis_stay"], "stay_id"].nunique()
            for h in horizons:
                r[f"pos_rows_{h}h"] += int(ps[f"label_sepsis_{h}h"].sum())
            if len(steps):
                r["treatment_steps"] += int((steps["split"] == s).sum())
        del df, pred, steps
        log.info("finalize: batch %d written (%d/%d)", k, i, len(batches))

    for s in SPLITS:
        r = rep[s]
        r["prevalence_6h_pct"] = round(100 * r["pos_rows_6h"] / max(r["pred_rows"], 1), 3) if 6 in horizons else None
    tot_sep, tot_all = sum(r["sepsis_stays"] for r in rep.values()), sum(r["stays"] for r in rep.values())
    if tot_sep == 0 or tot_sep == tot_all:
        issues.append(f"dataset has only one class (sepsis stays={tot_sep} of {tot_all})")
    comorb_report = None
    if comorb is not None and steps_seen:
        st = pd.concat(steps_seen, ignore_index=True)
        if st["stay_id"].duplicated().any():
            issues.append("a treatment stay appears in more than one batch")
        comorb_report = {"treatment_stays": int(len(st)),
                         "prevalence_pct_overall": {c: round(100 * float(st[c].mean()), 3) for c in STATE_COMORBIDITY_COLS},
                         "prevalence_pct_by_split": {sp: {c: round(100 * float(g[c].mean()), 3) for c in STATE_COMORBIDITY_COLS}
                                                     for sp, g in st.groupby("split")}}
    report = {"files_written_per_folder": len(batches), "comorbidities_in_treatment": comorb_report, "file_format": fmt, "splits": rep, "per_batch": per_batch,
              "validation_issues": sorted(set(issues)), "validation_passed": not issues,
              "action_cutpoints": stats["action_cutpoints"], "never_measured_in_train": stats["missing_in_train"]}
    (out_dir / "data_report.json").write_text(json.dumps(report, indent=2))
    return report
