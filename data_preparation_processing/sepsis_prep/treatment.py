"""Treatment-learning table: for SEPSIS stays only, one row per 4-hour decision step.

state       = patient features at the END of the previous block (everything known when deciding)
action      = what was actually given during this block: IV fluid (mL) and vasopressor (NE-equiv dose),
              each cut into bins 0..n-1  -> action = fluid_bin * n + vaso_bin  (5 x 5 = 25 actions)
reward      = SOFA_before - SOFA_after  (+ terminal_reward if survived / - if died, on the last step)
next_state  = features at the end of this block
context     = 6 static comorbidity flags (s_diabetes ... s_immunosuppression, 0/1), identical on every row of a stay,
              merged AFTER action / reward / done are built, so they can never change them
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .comorbidity import NEXT_COMORBIDITY_COLS, STATE_COMORBIDITY_COLS
from .features import CLINICAL

STATE_COLS = CLINICAL + ["age", "sex_male", "weight_kg", "careunit_code", "sofa_24h", "mech_vent", "urine_ml", "vaso_on"]


def _blocks(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    t = cfg["treatment"]
    d = df[df["is_sepsis"]].copy()
    d["hso"] = -d["hours_to_onset"].astype("float64")          # hours since onset (row end)
    d = d[(d["hso"] > -t["window_before_onset_h"]) & (d["hso"] <= t["window_after_onset_h"])]
    d["blk"] = (np.ceil(d["hso"] / t["step_hours"]) - 1).astype(int)
    return d.sort_values(["stay_id", "timestamp"])


def block_actions(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = _blocks(df, cfg)
    return d.groupby(["stay_id", "blk"], as_index=False).agg(
        fluid_block=("fluid_ml", "sum"), vaso_block=("vaso_ne_equiv", "max"))


def action_cutpoints(train_blocks: pd.DataFrame, n_bins: int) -> dict:
    qs = np.linspace(0, 1, n_bins)[1:-1]
    out = {}
    for name, col in (("fluid", "fluid_block"), ("vaso", "vaso_block")):
        v = train_blocks.loc[train_blocks[col] > 0, col]
        out[name] = [float(x) for x in v.quantile(qs)] if len(v) else [0.0] * len(qs)
    return out


def _bin(values: pd.Series, cuts: list[float]) -> pd.Series:
    b = 1 + np.digitize(values.values, cuts)
    return pd.Series(np.where(values.values > 0, b, 0), index=values.index).astype("int8")


def attach_comorbidities(steps: pd.DataFrame, comorb: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add the 6 static flags as RL context columns. Guarantees: same rows, same order, existing columns untouched,
    strictly 0/1, one value per stay. Key = stay_id (which was itself derived from subject_id + hadm_id upstream)."""
    if steps.empty:
        return steps
    if comorb is None:
        raise ValueError("comorbidities are enabled but no comorbidity table was given - run: python run.py comorbidities")
    miss = [c for c in STATE_COMORBIDITY_COLS if c not in comorb.columns]
    if miss:
        raise ValueError(f"comorbidity table lacks columns {miss}")
    ref = steps.copy()
    out = steps.merge(comorb[["stay_id"] + STATE_COMORBIDITY_COLS], on="stay_id", how="left", validate="m:1")
    if len(out) != len(ref):
        raise AssertionError("comorbidity merge changed the number of treatment rows")
    if out[STATE_COMORBIDITY_COLS].isna().any().any():
        bad = out.loc[out[STATE_COMORBIDITY_COLS].isna().any(axis=1), "stay_id"].unique()[:5].tolist()
        raise ValueError(f"treatment stays without a comorbidity row (e.g. {bad}) - cohort and comorbidity table differ")
    for c in STATE_COMORBIDITY_COLS:
        out[c] = out[c].astype("int8")
    if cfg.get("comorbidities", {}).get("add_next_state_columns", True):
        for s_col, ns_col in zip(STATE_COMORBIDITY_COLS, NEXT_COMORBIDITY_COLS):
            out[ns_col] = out[s_col]
    if not out[list(ref.columns)].reset_index(drop=True).equals(ref.reset_index(drop=True)):
        raise AssertionError("comorbidity merge altered existing treatment columns")
    return out


def build_steps(df: pd.DataFrame, cfg: dict, cuts: dict, comorb: pd.DataFrame | None = None) -> pd.DataFrame:
    t = cfg["treatment"]
    n = int(t["n_bins"])
    d = _blocks(df, cfg)
    if d.empty:
        return pd.DataFrame()
    act = block_actions(df, cfg)
    last = d.groupby(["stay_id", "blk"]).tail(1)
    keep = ["stay_id", "blk", "subject_id", "hadm_id", "batch_id", "split", "hospital_expire_flag", "hso"] + STATE_COLS
    bnd = last[keep]
    prev = bnd.copy()
    prev["blk"] = prev["blk"] + 1                                   # state of block k = end of block k-1
    st = prev.rename(columns={c: f"s_{c}" for c in STATE_COLS + ["hso"]}).rename(columns={"hso": "s_hours_since_onset"})
    st = st.rename(columns={"s_hso": "s_hours_since_onset"})
    nx = bnd[["stay_id", "blk"] + STATE_COLS + ["hso"]].rename(columns={c: f"ns_{c}" for c in STATE_COLS + ["hso"]})
    nx = nx.rename(columns={"ns_hso": "ns_hours_since_onset"})
    steps = act.merge(st, on=["stay_id", "blk"], how="inner").merge(nx, on=["stay_id", "blk"], how="inner")
    steps = steps.sort_values(["stay_id", "blk"]).reset_index(drop=True)

    steps["action_fluid_bin"] = _bin(steps["fluid_block"], cuts["fluid"])
    steps["action_vaso_bin"] = _bin(steps["vaso_block"], cuts["vaso"])
    steps["action"] = (steps["action_fluid_bin"].astype("int16") * n + steps["action_vaso_bin"]).astype("int16")
    steps["done"] = (steps.groupby("stay_id")["blk"].transform("max") == steps["blk"]).astype("int8")
    r = t["sofa_reward_scale"] * (steps["s_sofa_24h"] - steps["ns_sofa_24h"])
    died = steps["hospital_expire_flag"].astype(float) > 0
    term = np.where(died, -t["terminal_reward"], t["terminal_reward"])
    steps["reward"] = (r + np.where(steps["done"] == 1, term, 0.0)).astype("float32")
    steps = steps.rename(columns={"blk": "step"})
    for c in steps.columns:
        if steps[c].dtype == "float64":
            steps[c] = steps[c].astype("float32")
    if cfg.get("comorbidities", {}).get("enabled", False):
        steps = attach_comorbidities(steps, comorb, cfg)
    return steps
