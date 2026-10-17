#!/usr/bin/env python
"""Sepsis data preparation - one command line for everything.

  python run.py check        free: login + validates every SQL query (dry run, no billing)
  python run.py audit-items  print what the fluid / vasopressor item IDs really are
  python run.py cohort       step 1: choose sepsis + non-sepsis stays
  python run.py comorbidities  step 1b: 6 comorbidity flags per stay from diagnoses_icd (tiny query, needs only the cohort)
  python run.py estimate     free: how many GB will extract read from BigQuery
  python run.py extract      step 2: hourly data, batch by batch (resumable - just run again)
  python run.py finalize     step 3: impute + write output/early_prediction + output/treatment_recommendation (32 files each)
  python run.py all          cohort + extract + finalize
  python run.py smoke        tiny end-to-end test (200 stays, 2 batches) in output/smoke
  python run.py status       how many batches are done
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from sepsis_prep import sql as S
from sepsis_prep.bq import BQ
from sepsis_prep.cohort import build_cohort
from sepsis_prep.comorbidity import build_stay_comorbidities, validate_stay_table
from sepsis_prep.finalize import finalize
from sepsis_prep.hourly import extract_batch
from sepsis_prep.features import LAB_TABLES

log = logging.getLogger("sepsis_prep")


def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "run.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def batches_of(cohort: pd.DataFrame):
    """(batch number, stays of that batch) - the batch numbers were fixed when the cohort was built."""
    for k, b in cohort.groupby("batch_id", sort=True):
        yield int(k), b


def work_dir(cfg: dict, out: Path) -> Path:
    return out / cfg["paths"].get("work_dir", "_work")


def dummy_params(name: str, cfg: dict) -> dict:
    t0, t1 = datetime(2150, 1, 1), datetime(2150, 1, 2)
    p = {"stay_ids": [30000000], "t_start": t0, "t_end": t1}
    if name in LAB_TABLES:
        p = {"hadm_ids": [20000000], "lab_start": t0, "t_end": t1}
    if name == "inputevents":
        p["item_ids"] = list(cfg["fluid_itemids"]) + [int(v) for v in cfg["vasopressor_itemids"].values()]
    if name == "diagnoses":
        p = {"hadm_ids": [20000000], "subject_ids": [10000000]}
    return p


def cmd_check(cfg, bq, out):
    ok = True
    log.info("Checking login + every query (dry run = free) ...")
    queries = {"cohort": (S.cohort_sql(bq), {"min_age": 18}), **{k: (v, dummy_params(k, cfg)) for k, v in S.all_batch_queries(bq).items()},
               "diagnoses": (S.diagnoses_sql(bq), dummy_params("diagnoses", cfg))}
    for name, (sql, params) in queries.items():
        try:
            gb = bq.query_df(sql, params, label=name, dry_run=True)
            log.info("  OK    %-22s would scan %.2f GB", name, gb)
        except Exception as e:  # noqa: BLE001
            ok = False
            log.info("  FAIL  %-22s %s", name, str(e).splitlines()[0][:200])
    if ok:
        log.info("ALL GOOD. Next: python run.py cohort")
    else:
        log.info("Some queries failed. If every one says 'Access Denied' / 'credentials' -> run the gcloud login in README step 3.")
    return 0 if ok else 1


def cmd_audit(cfg, bq, out):
    ids = list(cfg["fluid_itemids"]) + [int(v) for v in cfg["vasopressor_itemids"].values()]
    df = bq.query_df(S.d_items_sql(bq), {"item_ids": ids}, label="d_items")
    fl = set(cfg["fluid_itemids"])
    df["used_as"] = ["FLUID" if i in fl else "VASOPRESSOR" for i in df["itemid"]]
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    log.info("\n%s", df.to_string(index=False))
    log.info("Check every FLUID label is really an IV fluid. Remove wrong IDs in config.yaml (fluid_itemids).")
    return 0


def cmd_cohort(cfg, bq, out):
    build_cohort(bq, cfg, work_dir(cfg, out))
    log.info("cohort cost so far: %.2f GB", bq.total_gb())
    return 0


def load_cohort(out: Path, cfg: dict) -> pd.DataFrame:
    p = work_dir(cfg, out) / "cohort.parquet"
    if not p.exists():
        raise SystemExit("cohort.parquet not found. Run: python run.py cohort")
    return pd.read_parquet(p)


def cmd_comorbidities(cfg, bq, out):
    """Cohort-wide, independent of the 32 raw batches: writes _work/comorbidities.parquet (one row per stay)."""
    wd = work_dir(cfg, out)
    cohort = load_cohort(out, cfg)
    n = int(cfg.get("comorbidities", {}).get("chunk_size", 10000))
    pairs = cohort[["subject_id", "hadm_id"]].drop_duplicates()
    parts = []
    for i in range(0, len(pairs), n):
        ch = pairs.iloc[i:i + n]
        parts.append(bq.query_df(S.diagnoses_sql(bq), {"hadm_ids": ch["hadm_id"].tolist(), "subject_ids": ch["subject_id"].unique().tolist()},
                                 label=f"diagnoses {i // n + 1}"))
    dx = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["subject_id", "hadm_id", "icd_version", "icd_code"])
    dx = dx.drop_duplicates()                      # chunks can overlap on subject_id lists; (subject, hadm) join removes the rest
    tbl, rep = build_stay_comorbidities(dx, cohort)
    issues = validate_stay_table(tbl)
    if issues:
        raise RuntimeError("comorbidity table invalid: " + "; ".join(issues))
    tbl.to_parquet(wd / "comorbidities.parquet", index=False)
    (wd / "comorbidity_report.json").write_text(json.dumps(rep, indent=2))
    log.info("comorbidities: %d stays, %d without any diagnosis row, prevalence %% = %s",
             rep["n_stays"], rep["stays_without_any_diagnosis_row"], rep["prevalence_pct"])
    return 0


def cmd_estimate(cfg, bq, out):
    cohort = load_cohort(out, cfg)
    n = int(cohort["batch_id"].nunique())
    first = next(batches_of(cohort))[1]
    base = {"stay_ids": first["stay_id"].tolist(), "t_start": first["window_start"].min(),
            "t_end": first["window_end"].max() + pd.Timedelta(hours=1)}
    tot = 0.0
    for name, sql in S.all_batch_queries(bq).items():
        p = dict(base)
        if name in LAB_TABLES:
            p = {"hadm_ids": first["hadm_id"].unique().tolist(), "lab_start": base["t_start"] - pd.Timedelta(hours=24), "t_end": base["t_end"]}
        if name == "inputevents":
            p["item_ids"] = list(cfg["fluid_itemids"]) + [int(v) for v in cfg["vasopressor_itemids"].values()]
        tot += bq.query_df(sql, p, label=name, dry_run=True)
    log.info("One batch reads ~%.1f GB. %d batches -> ~%.0f GB total. (BigQuery: first 1 TB/month is free.)", tot, n, tot * n)
    log.info("Want it cheaper? use fewer batches (hourly.n_batches in config.yaml) - but each batch then needs more RAM.")
    return 0


def cmd_extract(cfg, bq, out, max_batches=None):
    wd = work_dir(cfg, out)
    cohort = load_cohort(out, cfg)
    rep = json.loads((wd / "cohort_report.json").read_text())
    bdir = wd / "raw_batches"
    bdir.mkdir(parents=True, exist_ok=True)
    hfile = bdir / "_cohort_hash.txt"
    if hfile.exists() and hfile.read_text().strip() != rep["cohort_hash"]:
        raise SystemExit("The cohort changed since the batches were made. Delete the folder "
                         f"{bdir} and run extract again (or restore the old config).")
    hfile.write_text(rep["cohort_hash"])
    all_batches = list(batches_of(cohort))
    total = len(all_batches)
    todo = [(k, b) for k, b in all_batches if not (bdir / f"batch_{k:02d}.parquet").exists()]
    log.info("Step 2/3  extract: %d batches total, %d already done, %d to do", total, total - len(todo), len(todo))
    if max_batches:
        todo = todo[:max_batches]
    t0 = time.time()
    for n, (k, b) in enumerate(todo, 1):
        ts = time.time()
        df = extract_batch(bq, b, cfg)
        tmp = bdir / f"batch_{k:02d}.parquet.tmp"
        df.to_parquet(tmp, index=False)
        tmp.replace(bdir / f"batch_{k:02d}.parquet")      # only finished batches ever have the final name
        eta = (time.time() - t0) / n * (len(todo) - n) / 60
        log.info("batch %d/%d done: %d stays, %d hourly rows, %.0fs  (BigQuery so far %.1f GB, ETA %.0f min)",
                 k, total, len(b), len(df), time.time() - ts, bq.total_gb(), eta)
        del df
        gc.collect()
    left = total - len(list(bdir.glob("batch_*.parquet")))
    log.info("extract finished. batches still missing: %d", left)
    return 0 if left == 0 else 2


def cmd_finalize(cfg, bq, out):
    rep = finalize(cfg, out)
    log.info(json.dumps(rep["splits"], indent=2))
    if rep["validation_passed"]:
        log.info("VALIDATION PASSED.\n  early prediction      : %s\n  treatment recommendation: %s",
                 out / cfg["paths"]["early_dir"], out / cfg["paths"]["treatment_dir"])
    else:
        log.info("VALIDATION PROBLEMS: %s", rep["validation_issues"])
    return 0 if rep["validation_passed"] else 1


def cmd_status(cfg, bq, out):
    wd = work_dir(cfg, out)
    done = len(list((wd / "raw_batches").glob("batch_*.parquet"))) if (wd / "raw_batches").exists() else 0
    try:
        total = int(load_cohort(out, cfg)["batch_id"].nunique())
    except SystemExit:
        total = "?"
    e, tr = out / cfg["paths"]["early_dir"], out / cfg["paths"]["treatment_dir"]
    n_e = len(list(e.glob("batch_*"))) if e.exists() else 0
    n_t = len(list(tr.glob("batch_*"))) if tr.exists() else 0
    log.info("extracted batches: %s / %s ; final files: early_prediction=%d, treatment_recommendation=%d", done, total, n_e, n_t)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "audit-items", "cohort", "comorbidities", "estimate", "extract", "finalize", "all", "smoke", "status"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-batches", type=int, default=None, help="extract only N new batches, then stop")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.command == "smoke":
        cfg = copy.deepcopy(cfg)
        cfg["cohort"]["limit_stays"] = 200
        cfg["hourly"]["n_batches"] = 2
        out = Path(cfg["paths"]["out_dir"]) / "smoke"
    else:
        out = Path(cfg["paths"]["out_dir"])
    setup_logging(work_dir(cfg, out))
    bq = BQ(cfg)
    try:
        if a.command == "all" or a.command == "smoke":
            cmd_cohort(cfg, bq, out)
            if cfg.get("comorbidities", {}).get("enabled", False):
                cmd_comorbidities(cfg, bq, out)
            rc = cmd_extract(cfg, bq, out)
            if rc == 0:
                rc = cmd_finalize(cfg, bq, out)
            return rc
        fn = {"check": cmd_check, "audit-items": cmd_audit, "cohort": cmd_cohort, "comorbidities": cmd_comorbidities, "estimate": cmd_estimate,
              "finalize": cmd_finalize, "status": cmd_status}.get(a.command)
        return fn(cfg, bq, out) if fn else cmd_extract(cfg, bq, out, a.max_batches)
    except Exception as e:  # noqa: BLE001
        log.exception("STOPPED: %s", e)
        log.info("Nothing is lost - finished batches are saved. Fix the problem and run the same command again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
