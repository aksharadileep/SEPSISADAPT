"""Step 2: one batch of stays -> hourly table (1 row = 1 stay x 1 hour).

Rules that prevent leakage
  * row `timestamp` = END of the hour; it only contains information measured before that moment
  * values are carried forward only (never backward), and only up to a limit (config: ffill_hours)
  * label_sepsis_Hh = 1  iff sepsis onset is AFTER this row and within H hours
  * rows with timestamp >= onset are marked already_septic (used for treatment learning, never for prediction)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import sql as S
from .features import FEATURES, LAB_TABLES, label_cols

log = logging.getLogger("sepsis_prep")
H1 = np.timedelta64(1, "h")


def _us(s):
    return pd.to_datetime(s).astype("datetime64[us]")


# --------------------------------------------------------------------------- grid + labels
def build_grid(c: pd.DataFrame, horizons) -> pd.DataFrame:
    n = np.ceil((c["window_end"] - c["window_start"]).dt.total_seconds() / 3600.0).clip(lower=1).astype(int).values
    rep = np.repeat(np.arange(len(c)), n)
    hour = np.concatenate([np.arange(k) for k in n])
    g = c.iloc[rep].reset_index(drop=True)
    g["hour"] = hour.astype("int32")
    g["timestamp"] = _us(g["window_start"] + pd.to_timedelta(hour + 1, unit="h"))
    g["hours_since_icu_admission"] = ((g["timestamp"] - g["intime"]).dt.total_seconds() / 3600.0).astype("float32")
    onset = g["sepsis_onset_time"]
    has = onset.notna()
    g["hours_to_onset"] = ((onset - g["timestamp"]).dt.total_seconds() / 3600.0).astype("float32")  # >0 = onset in future
    g["already_septic"] = (has & (g["timestamp"] >= onset)).astype("int8")
    for h in horizons:
        g[f"label_sepsis_{h}h"] = (has & (g["hours_to_onset"] > 0) & (g["hours_to_onset"] <= h)).astype("int8")
    return g


# --------------------------------------------------------------------------- carry-forward
def attach_asof(grid: pd.DataFrame, obs: pd.DataFrame | None, name: str, limit_h: float, flags: bool) -> pd.DataFrame:
    """obs: stay_id, hour_end, value. Backward as-of join with a staleness limit."""
    if obs is None or len(obs) == 0:
        grid[name] = np.float32(np.nan)
        if flags:
            grid[f"{name}_obs"] = np.int8(0)
        return grid
    right = (obs.dropna(subset=["value"]).drop_duplicates(["stay_id", "hour_end"], keep="last")
             .rename(columns={"value": name, "hour_end": "_obs_t"})[["stay_id", "_obs_t", name]])
    right["_obs_t"] = _us(right["_obs_t"])
    right[name] = right[name].astype("float64")
    right = right.sort_values("_obs_t")
    out = pd.merge_asof(grid.sort_values("timestamp"), right, left_on="timestamp", right_on="_obs_t",
                        by="stay_id", direction="backward", tolerance=pd.Timedelta(hours=limit_h))
    if flags:
        out[f"{name}_obs"] = (out["_obs_t"] == out["timestamp"]).astype("int8")
    return out.drop(columns="_obs_t")


# --------------------------------------------------------------------------- interval -> hours
def _expand(df: pd.DataFrame):
    """Repeat every interval once per hour bucket it touches."""
    d = df.dropna(subset=["starttime"]).copy()
    d["starttime"], d["endtime"] = _us(d["starttime"]), _us(d["endtime"].fillna(d["starttime"]))
    d["endtime"] = d["endtime"].where(d["endtime"] >= d["starttime"], d["starttime"])
    s0, e0 = d["starttime"].dt.floor("h"), d["endtime"].dt.floor("h")
    nb = ((e0 - s0) / pd.Timedelta(hours=1)).astype(int).values + 1
    rep = np.repeat(np.arange(len(d)), nb)
    k = np.concatenate([np.arange(m) for m in nb]) if len(nb) else np.array([], dtype=int)
    x = d.iloc[rep].reset_index(drop=True)
    bs = s0.iloc[rep].values + k.astype("timedelta64[h]")
    seg_s = np.maximum(x["starttime"].values, bs)
    seg_e = np.minimum(x["endtime"].values, bs + H1)
    x["_overlap_s"] = (seg_e - seg_s) / np.timedelta64(1, "s")
    x["_dur_s"] = (x["endtime"] - x["starttime"]).dt.total_seconds()
    x["hour_end"] = _us(pd.Series(bs + H1))
    return x


def spread_amount(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Total amount (mL) split over hours proportionally to time overlap."""
    if df is None or len(df) == 0:
        return pd.DataFrame({"stay_id": pd.Series(dtype="int64"), "hour_end": pd.Series(dtype="datetime64[us]"), "out": pd.Series(dtype="float64")})
    x = _expand(df.dropna(subset=[col]))
    x["_c"] = np.where(x["_dur_s"] <= 0, x[col], x[col] * x["_overlap_s"] / x["_dur_s"].replace(0, np.nan))
    return x.groupby(["stay_id", "hour_end"], as_index=False)["_c"].sum().rename(columns={"_c": "out"})


def spread_rate(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Time-weighted mean rate inside each hour."""
    if df is None or len(df) == 0:
        return pd.DataFrame({"stay_id": pd.Series(dtype="int64"), "hour_end": pd.Series(dtype="datetime64[us]"), "out": pd.Series(dtype="float64")})
    x = _expand(df.dropna(subset=[col]))
    x["_c"] = np.where(x["_dur_s"] <= 0, 0.0, x[col] * x["_overlap_s"] / 3600.0)
    return x.groupby(["stay_id", "hour_end"], as_index=False)["_c"].sum().rename(columns={"_c": "out"})


def flag_hours(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame({"stay_id": pd.Series(dtype="int64"), "hour_end": pd.Series(dtype="datetime64[us]"), "out": pd.Series(dtype="float64")})
    x = _expand(df)
    return x[["stay_id", "hour_end"]].drop_duplicates().assign(out=1.0)


def ne_equivalent(ie: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Raw vasopressor infusion rows -> one rate in norepinephrine-equivalent mcg/kg/min."""
    ids = {int(v): k for k, v in cfg["vasopressor_itemids"].items()}
    v = ie[ie["itemid"].isin(ids)].copy()
    if v.empty:
        return v.assign(ne_rate=pd.Series(dtype="float64"))
    agent = v["itemid"].map(ids)
    rate = v["rate"].astype("float64")
    unit = v["rateuom"].fillna("").str.lower()
    wt = v["patientweight"].astype("float64").where(lambda s: s > 0)
    rate = np.where(unit == "mcg/min", rate / wt, rate)          # make it per kg
    rate = np.where(unit == "ng/kg/min", rate / 1000.0, rate)    # ng -> mcg
    rate = np.where(unit == "units/hour", rate / 60.0, rate)     # vasopressin -> units/min
    v["ne_rate"] = pd.Series(rate, index=v.index) * agent.map(cfg["vasopressor_factors"]).astype("float64")
    return v


# --------------------------------------------------------------------------- assemble
def _stay_keyed(df, key_map):
    if df is None or len(df) == 0:
        return df
    if "stay_id" in df.columns:
        return df
    return df.merge(key_map, on="hadm_id", how="inner")


def _clip(series: pd.Series, name: str, ranges: dict) -> pd.Series:
    if name in ranges:
        lo, hi = ranges[name]
        return series.where((series >= lo) & (series <= hi))
    return series


def assemble_batch(c: pd.DataFrame, src: dict, cfg: dict) -> pd.DataFrame:
    hc = cfg["hourly"]
    flags = bool(hc.get("add_obs_flags", True))
    lim = hc["ffill_hours"]
    ranges = cfg.get("plausible_ranges", {})
    grid = build_grid(c, hc["horizons"])
    key_map = c[["hadm_id", "stay_id"]].drop_duplicates()

    # ---- measured variables (vitals + labs): carry forward with limit
    for feat, (source, fam) in FEATURES.items():
        df = _stay_keyed(src.get(source), key_map)
        if df is None or len(df) == 0 or feat not in df.columns:
            obs = None
        else:
            obs = df[["stay_id", "hour_end", feat]].rename(columns={feat: "value"})
            obs = obs.assign(value=_clip(obs["value"].astype("float64"), feat, ranges))
        grid = attach_asof(grid, obs, feat, lim[fam], flags)

    # ---- SOFA (sampled at the same hour, tolerance 2h)
    sofa = src.get("sofa")
    obs = None if sofa is None or len(sofa) == 0 else sofa.rename(columns={"sofa_24h": "value"})[["stay_id", "hour_end", "value"]]
    grid = attach_asof(grid, obs, "sofa_24h", 2, False)

    # ---- treatments
    ie = src.get("inputevents")
    if ie is None or len(ie) == 0:
        fl = spread_amount(None, "amount"); va = spread_rate(None, "ne_rate")
        weight = pd.DataFrame({"stay_id": pd.Series(dtype="int64"), "weight_kg": pd.Series(dtype="float64")})
    else:
        fl = spread_amount(ie[ie["itemid"].isin(cfg["fluid_itemids"]) & (ie["amountuom"] == "mL")], "amount")
        va = spread_rate(ne_equivalent(ie, cfg), "ne_rate")
        w = ie.dropna(subset=["patientweight"]).sort_values("starttime").drop_duplicates("stay_id")
        weight = w[["stay_id", "patientweight"]].rename(columns={"patientweight": "weight_kg"})
    grid = grid.merge(fl.rename(columns={"out": "fluid_ml"}), left_on=["stay_id", "timestamp"], right_on=["stay_id", "hour_end"], how="left").drop(columns="hour_end")
    grid = grid.merge(va.rename(columns={"out": "vaso_ne_equiv"}), left_on=["stay_id", "timestamp"], right_on=["stay_id", "hour_end"], how="left").drop(columns="hour_end")
    grid["fluid_ml"] = grid["fluid_ml"].fillna(0.0).astype("float32")
    grid["vaso_ne_equiv"] = grid["vaso_ne_equiv"].fillna(0.0).astype("float32")
    grid["vaso_on"] = (grid["vaso_ne_equiv"] > 0).astype("int8")

    vent = src.get("ventilation")
    if vent is not None and len(vent):
        vv = vent[vent["ventilation_status"].isin(["InvasiveVent", "Tracheostomy"])]
        vh = flag_hours(vv).rename(columns={"out": "mech_vent"})
    else:
        vh = flag_hours(None).rename(columns={"out": "mech_vent"})
    grid = grid.merge(vh, left_on=["stay_id", "timestamp"], right_on=["stay_id", "hour_end"], how="left").drop(columns="hour_end")
    grid["mech_vent"] = grid["mech_vent"].fillna(0).astype("int8")

    uo = src.get("urine")
    if uo is not None and len(uo):
        uo = uo.rename(columns={"urine_ml": "urine_ml"})[["stay_id", "hour_end", "urine_ml"]].dropna()
        grid = grid.merge(uo, left_on=["stay_id", "timestamp"], right_on=["stay_id", "hour_end"], how="left").drop(columns="hour_end")
    else:
        grid["urine_ml"] = np.nan
    grid["urine_obs"] = grid["urine_ml"].notna().astype("int8")
    grid["urine_ml"] = grid["urine_ml"].fillna(0.0).astype("float32")

    grid = grid.merge(weight, on="stay_id", how="left")
    grid["weight_kg"] = _clip(grid["weight_kg"].astype("float64"), "weight_kg", {"weight_kg": [20, 400]})

    for f in list(FEATURES) + ["weight_kg", "sofa_24h"]:
        grid[f] = grid[f].astype("float32")
    return grid.sort_values(["stay_id", "hour"]).reset_index(drop=True)


# --------------------------------------------------------------------------- extract (talks to BigQuery)
def extract_batch(bq, c: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    stay_ids = c["stay_id"].tolist()
    hadm_ids = c["hadm_id"].unique().tolist()
    t_start = c["window_start"].min()
    t_end = c["window_end"].max() + pd.Timedelta(hours=1)
    base = {"stay_ids": stay_ids, "t_start": t_start, "t_end": t_end}
    q = S.all_batch_queries(bq)
    src = {}
    for name, sql in q.items():
        p = dict(base)
        if name in LAB_TABLES:
            p = {"hadm_ids": hadm_ids, "lab_start": t_start - pd.Timedelta(hours=24), "t_end": t_end}
        if name == "inputevents":
            ids = list(cfg["fluid_itemids"]) + [int(v) for v in cfg["vasopressor_itemids"].values()]
            p["item_ids"] = ids
        src[name] = bq.query_df(sql, p, label=name)
    return assemble_batch(c, src, cfg)
