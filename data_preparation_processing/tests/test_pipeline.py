"""Offline tests with synthetic data (no BigQuery needed).  Run:  python -m pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shutil

from sepsis_prep import comorbidity as CM
from sepsis_prep import hourly as H
from sepsis_prep import treatment as T
from sepsis_prep.cohort import assign_batches, refine_cohort, split_of
from sepsis_prep.features import LAB_TABLES, VITALS
from sepsis_prep.finalize import finalize
from sepsis_prep.hourly import extract_batch

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
T0 = pd.Timestamp("2150-03-01 08:20:00")


def raw_cohort(n_sep=30, n_ctl=60, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sep + n_ctl):
        sep = i < n_sep
        intime = T0 + pd.Timedelta(hours=int(rng.integers(0, 500)), minutes=int(rng.integers(0, 60)))
        los = float(rng.integers(40, 120))
        onset = intime + pd.Timedelta(hours=float(rng.integers(8, 36)), minutes=17)
        rows.append(dict(subject_id=1000 + i, hadm_id=2000 + i, stay_id=3000 + i, first_careunit="MICU" if i % 2 else "SICU",
                         intime=intime, outtime=intime + pd.Timedelta(hours=los), los_hours=los, stay_rank=1,
                         gender="M" if i % 3 else "F", age=50 + i % 30, hospital_expire_flag=int(i % 5 == 0),
                         suspected_infection_time=onset - pd.Timedelta(hours=2) if sep else pd.NaT,
                         sofa_time=onset if sep else pd.NaT, is_sepsis=sep))
    return pd.DataFrame(rows)


# one known disease per stay (cycle of 6) + a noise code + some stays with NO diagnosis rows at all
_DX = [(10, "E119"), (9, "42843"), (10, "N189"), (10, "K7469"), (10, "C349"), (10, "B20")]


def make_dx(cohort):
    rows, expect = [], {}
    for i, r in enumerate(cohort.itertuples()):
        flags = dict.fromkeys(CM.STATE_COMORBIDITY_COLS, 0)
        if i % 7 != 0:                                               # stays 0,7,14.. have no diagnosis rows
            v, code = _DX[i % 6]
            rows += [dict(subject_id=r.subject_id, hadm_id=r.hadm_id, icd_version=v, icd_code=code),
                     dict(subject_id=r.subject_id, hadm_id=r.hadm_id, icd_version=10, icd_code="I10"),     # noise
                     dict(subject_id=r.subject_id, hadm_id=r.hadm_id, icd_version=9, icd_code="4019")]     # noise
            flags[CM.STATE_COMORBIDITY_COLS[i % 6]] = 1
        expect[r.stay_id] = flags
    return pd.DataFrame(rows), expect


def write_comorb(tmp_path, cfg, cohort):
    dx, expect = make_dx(cohort)
    tbl, _ = CM.build_stay_comorbidities(dx, cohort)
    wd = tmp_path / cfg["paths"]["work_dir"]
    wd.mkdir(parents=True, exist_ok=True)
    tbl.to_parquet(wd / "comorbidities.parquet", index=False)
    return expect


class FakeBQ:
    """Returns synthetic tables in the same shape as the real SQL results."""
    HOSP, ICU, DERIVED = "h", "i", "d"

    def __init__(self, cohort):
        self.c = cohort
        self.rng = np.random.default_rng(5)
        self.log = []

    def total_gb(self):
        return 0.0

    def query_df(self, sql, params=None, label="q", dry_run=False, **kw):
        c, rng = self.c, self.rng
        sid = set(params.get("stay_ids", []))
        cc = c[c["stay_id"].isin(sid)] if sid else c[c["hadm_id"].isin(params.get("hadm_ids", []))]
        if label in ("vitals", "gcs"):
            rows = []
            for r in cc.itertuples():
                for h in range(0, int((r.window_end - r.window_start).total_seconds() // 3600) + 1):
                    if rng.random() < 0.7:
                        d = dict(stay_id=r.stay_id, hour_end=r.window_start + pd.Timedelta(hours=h + 1))
                        if label == "vitals":
                            d.update({v: 80 + rng.normal() * 5 for v in VITALS})
                            d["heart_rate"] = 999 if rng.random() < 0.02 else d["heart_rate"]   # garbage value
                        else:
                            d["gcs"] = 14.0
                        rows.append(d)
            return pd.DataFrame(rows)
        if label in LAB_TABLES:
            rows = []
            for r in cc.itertuples():
                for h in range(0, 60, 7):
                    d = dict(hadm_id=r.hadm_id, hour_end=r.window_start + pd.Timedelta(hours=h + 1))
                    d.update({k: 1.0 + rng.random() for k in LAB_TABLES[label]})
                    rows.append(d)
            return pd.DataFrame(rows)
        if label == "urine":
            return pd.DataFrame([dict(stay_id=r.stay_id, hour_end=r.window_start + pd.Timedelta(hours=h + 1), urine_ml=40.0)
                                 for r in cc.itertuples() for h in range(0, 30, 2)])
        if label == "inputevents":
            rows = []
            for r in cc.itertuples():
                st = r.intime + pd.Timedelta(hours=3)
                rows.append(dict(stay_id=r.stay_id, starttime=st, endtime=st + pd.Timedelta(hours=2), itemid=225158, amount=1000.0,
                                 amountuom="mL", rate=500.0, rateuom="mL/hour", patientweight=80.0))
                if r.is_sepsis:
                    st = r.sepsis_onset_time + pd.Timedelta(hours=1)
                    rows.append(dict(stay_id=r.stay_id, starttime=st, endtime=st + pd.Timedelta(hours=5), itemid=221906, amount=1.0,
                                     amountuom="mg", rate=0.2, rateuom="mcg/kg/min", patientweight=80.0))
                    rows.append(dict(stay_id=r.stay_id, starttime=st + pd.Timedelta(hours=1), endtime=st + pd.Timedelta(hours=9), itemid=225158,
                                     amount=800.0, amountuom="mL", rate=100.0, rateuom="mL/hour", patientweight=80.0))
            return pd.DataFrame(rows)
        if label == "ventilation":
            return pd.DataFrame([dict(stay_id=r.stay_id, starttime=r.intime + pd.Timedelta(hours=1), endtime=r.intime + pd.Timedelta(hours=6),
                                      ventilation_status="InvasiveVent") for r in cc.itertuples()][:10])
        if label == "sofa":
            return pd.DataFrame([dict(stay_id=r.stay_id, hour_end=r.window_start + pd.Timedelta(hours=h + 1), sofa_24h=float(2 + h // 20))
                                 for r in cc.itertuples() for h in range(0, 100)])
        raise KeyError(label)


# ------------------------------------------------------------------ unit tests
def test_split_is_patient_level_and_deterministic():
    assert split_of(123, 42, .7, .15) == split_of(123, 42, .7, .15)
    s = [split_of(i, 42, .7, .15) for i in range(5000)]
    assert 0.66 < s.count("train") / 5000 < 0.74


def test_refine_cohort_onset_and_exclusions():
    raw = raw_cohort()
    raw.loc[0, "sofa_time"] = raw.loc[0, "intime"] + pd.Timedelta(hours=1)      # too early -> dropped
    raw.loc[0, "suspected_infection_time"] = raw.loc[0, "intime"] - pd.Timedelta(hours=5)
    out, rep = refine_cohort(raw, CFG)
    assert 3000 not in set(out["stay_id"])
    assert rep["sepsis_dropped_onset_before_or_too_soon_after_icu_admission"] == 1
    sep = out[out["is_sepsis"]]
    assert sep["sepsis_onset_time"].notna().all() and out.loc[~out["is_sepsis"], "sepsis_onset_time"].isna().all()
    assert (sep["window_end"] <= sep["sepsis_onset_time"] + pd.Timedelta(hours=48, seconds=1)).all()


def test_labels_only_before_onset():
    out, _ = refine_cohort(raw_cohort(), CFG)
    g = H.build_grid(out.head(10), [6, 12, 24])
    pos = g[g["label_sepsis_6h"] == 1]
    assert ((pos["hours_to_onset"] > 0) & (pos["hours_to_onset"] <= 6)).all()
    assert (g.loc[g["already_septic"] == 1, "label_sepsis_6h"] == 0).all()
    assert (g.loc[~g["is_sepsis"], "label_sepsis_6h"] == 0).all()
    assert g.groupby("stay_id")["label_sepsis_6h"].sum().loc[lambda s: s.index.isin(g[g.is_sepsis].stay_id)].between(1, 6).all()


def test_forward_fill_limit_never_looks_ahead():
    grid = pd.DataFrame({"stay_id": 1, "timestamp": pd.date_range("2150-01-01 01:00", periods=12, freq="h").astype("datetime64[us]")})
    obs = pd.DataFrame({"stay_id": [1], "hour_end": [pd.Timestamp("2150-01-01 03:00")], "value": [7.0]})
    out = H.attach_asof(grid, obs, "x", 3, True).sort_values("timestamp").reset_index(drop=True)
    assert out["x"].iloc[:2].isna().all()                       # before measurement -> NaN (no look-ahead)
    assert out["x"].iloc[2:6].tolist() == [7.0] * 4             # at obs hour + 3h carried
    assert out["x"].iloc[6:].isna().all()                       # stale -> NaN again
    assert out["x_obs"].tolist() == [0, 0, 1] + [0] * 9


def test_spread_amount_conserves_volume():
    df = pd.DataFrame({"stay_id": [1], "starttime": [pd.Timestamp("2150-01-01 10:30")], "endtime": [pd.Timestamp("2150-01-01 13:30")], "amount": [900.0]})
    out = H.spread_amount(df, "amount")
    assert out["out"].sum() == pytest.approx(900.0)
    assert out.sort_values("hour_end")["out"].round(1).tolist() == [150.0, 300.0, 300.0, 150.0]


def test_vasopressor_units():
    ie = pd.DataFrame({"itemid": [221906, 222315, 221662], "rate": [0.1, 60.0, 5.0], "rateuom": ["mcg/kg/min", "units/hour", "mcg/kg/min"],
                       "patientweight": [80.0] * 3})
    r = H.ne_equivalent(ie, CFG)["ne_rate"].tolist()
    assert r == pytest.approx([0.1, 2.5, 0.05])                 # 60 U/h = 1 U/min -> x2.5 ; dopamine x0.01


def test_action_bins():
    blocks = pd.DataFrame({"fluid_block": [0, 100, 200, 300, 400, 500, 600, 700], "vaso_block": [0, 0, 0, .1, .2, .3, .4, .5]})
    cuts = T.action_cutpoints(blocks, 5)
    b = T._bin(blocks["fluid_block"], cuts["fluid"])
    assert b.iloc[0] == 0 and b.max() == 4 and b.min() == 0 and b.iloc[1] == 1


def test_batches_are_balanced_and_mixed():
    is_sep = np.array([True] * 100 + [False] * 700)
    b = assign_batches(is_sep, 32, seed=1)
    assert set(b) == set(range(1, 33))
    sizes = np.bincount(b)[1:]
    assert sizes.max() - sizes.min() <= 1                       # all 32 batches (almost) the same size
    sep_per = np.bincount(b[is_sep], minlength=33)[1:]
    non_per = np.bincount(b[~is_sep], minlength=33)[1:]
    assert sep_per.min() >= 3 and non_per.min() >= 21           # every batch has BOTH kinds of patients
    assert (assign_batches(is_sep, 32, seed=1) == b).all()      # deterministic


# ------------------------------------------------------------------ end-to-end (fake BigQuery)
def test_end_to_end(tmp_path):
    cfg = json.loads(json.dumps(CFG))
    cfg["hourly"]["n_batches"] = 4
    cohort, rep = refine_cohort(raw_cohort(), cfg)
    assert rep["n_batches"] == 4 and rep["sepsis_stays_per_batch_min_max"][0] >= 1
    bq = FakeBQ(cohort)
    bdir = tmp_path / cfg["paths"]["work_dir"] / "raw_batches"
    bdir.mkdir(parents=True)
    for k, b in cohort.groupby("batch_id"):
        extract_batch(bq, b, cfg).to_parquet(bdir / f"batch_{int(k):02d}.parquet", index=False)
    expect = write_comorb(tmp_path, cfg, cohort)
    r = finalize(cfg, tmp_path)
    assert r["validation_passed"], r["validation_issues"]

    ed, td = tmp_path / cfg["paths"]["early_dir"], tmp_path / cfg["paths"]["treatment_dir"]
    assert sorted(p.name for p in ed.glob("batch_*")) == [f"batch_{k:02d}.parquet" for k in range(1, 5)]
    assert sorted(p.name for p in td.glob("batch_*")) == [f"batch_{k:02d}.parquet" for k in range(1, 5)]

    pred = pd.concat(pd.read_parquet(p) for p in sorted(ed.glob("batch_*.parquet")))
    assert pred["heart_rate"].max() < 300                      # garbage 999 was removed
    assert pred.groupby("subject_id")["split"].nunique().max() == 1
    assert pred["label_sepsis_6h"].sum() > 0 and pred["is_sepsis_stay"].any() and (~pred["is_sepsis_stay"]).any()
    for p_ in sorted(ed.glob("batch_*.parquet")):               # every batch file has sepsis AND non-sepsis patients
        one = pd.read_parquet(p_)
        assert one["is_sepsis_stay"].any() and (~one["is_sepsis_stay"]).any()
    assert pred[json.loads((tmp_path / "feature_columns.json").read_text())["model_inputs"]].isna().sum().sum() == 0
    # no row at/after onset in the prediction file
    assert (pred.loc[pred["is_sepsis_stay"], "hours_to_onset"] > 0).all()

    steps = pd.concat(pd.read_parquet(p) for p in sorted(td.glob("batch_*.parquet")))
    sepsis_ids = set(pred.loc[pred["is_sepsis_stay"], "stay_id"])
    assert len(steps) > 0 and steps["action"].between(0, 24).all()
    assert set(steps["stay_id"]) <= sepsis_ids                   # treatment file = sepsis patients only
    assert steps.groupby("stay_id")["done"].sum().eq(1).all()
    assert steps.loc[steps["done"] == 1, "reward"].abs().max() > 5   # terminal +/-15 present
    assert steps.filter(like="s_heart_rate").notna().all().all() and steps.filter(like="ns_heart_rate").notna().all().all()
    # state of step k+1 == next_state of step k
    s = steps.sort_values(["stay_id", "step"])
    nxt = s.groupby("stay_id")[["ns_heart_rate"]].shift(1)["ns_heart_rate"]
    cur = s["s_heart_rate"]
    ok = nxt.notna() & (s.groupby("stay_id")["step"].diff() == 1)
    assert np.allclose(nxt[ok], cur[ok])
    # --- comorbidities: values equal the independent expectation, constant per stay, 0/1
    for c in CM.STATE_COMORBIDITY_COLS:
        assert steps[c].isin([0, 1]).all() and steps[c].dtype == "int8"
    assert (steps.groupby("stay_id")[CM.STATE_COMORBIDITY_COLS + CM.NEXT_COMORBIDITY_COLS].nunique() == 1).all().all()
    got = steps.drop_duplicates("stay_id").set_index("stay_id")[CM.STATE_COMORBIDITY_COLS]
    for sid, row in got.iterrows():
        assert row.to_dict() == expect[sid], sid
    assert r["comorbidities_in_treatment"]["treatment_stays"] == steps["stay_id"].nunique()
    fc = json.loads((tmp_path / "feature_columns.json").read_text())
    assert fc["treatment_state_comorbidity"] == CM.STATE_COMORBIDITY_COLS
    assert (tmp_path / "data_report.json").exists() and len(r["per_batch"]) == 4


def test_csv_output(tmp_path):
    cfg = json.loads(json.dumps(CFG))
    cfg["hourly"]["n_batches"] = 2
    cfg["paths"]["file_format"] = "csv"
    cohort, _ = refine_cohort(raw_cohort(n_sep=10, n_ctl=20), cfg)
    bq = FakeBQ(cohort)
    bdir = tmp_path / cfg["paths"]["work_dir"] / "raw_batches"
    bdir.mkdir(parents=True)
    for k, b in cohort.groupby("batch_id"):
        extract_batch(bq, b, cfg).to_parquet(bdir / f"batch_{int(k):02d}.parquet", index=False)
    write_comorb(tmp_path, cfg, cohort)
    assert finalize(cfg, tmp_path)["validation_passed"]
    assert (tmp_path / cfg["paths"]["early_dir"] / "batch_01.csv").exists()


# ------------------------------------------------------------------ comorbidity tests
@pytest.mark.parametrize("name,version,pos,neg", [
    ("diabetes", 9, ["25000", "2500", "36641"], ["V4585", "6480", "2510"]),
    ("diabetes", 10, ["E119", "E1010", "E139"], ["R739", "O2411", "E669"]),
    ("heart_failure", 9, ["4280", "42843", "40491", "4254"], ["4019", "4140", "4251"]),
    ("heart_failure", 10, ["I5022", "I110", "I130", "I420", "I43"], ["I10", "I4891", "I2510"]),
    ("chronic_kidney_disease", 9, ["5859", "40391", "V4511", "V560"], ["5849", "586", "40300x"[:5] + "9"][:2]),
    ("chronic_kidney_disease", 10, ["N189", "N186", "I129", "Z992"], ["N179", "N19", "N390"]),
    ("chronic_liver_disease", 9, ["5715", "5712", "4560", "07054", "V427"], ["5711", "570", "5733"]),
    ("chronic_liver_disease", 10, ["K7469", "K703", "K7031", "B182", "I8501"], ["K701", "K7200", "K760"]),
    ("cancer", 9, ["1629", "19889", "2000", "20500", "1400"], ["173", "1739", "V1000", "2110", "2800"]),
    ("cancer", 10, ["C349", "C7951", "C9100", "C7A1", "C43"[:3]], ["C44", "C4491", "D369", "Z8579", "D45"]),
    ("immunosuppression", 9, ["042", "V08", "2793", "V420", "99681", "V5865"], ["2794", "2880", "V1000", "4019"]),
    ("immunosuppression", 10, ["B20", "Z21", "D849", "Z940", "Z7952", "T861", "D8981"], ["D709", "C349", "Z7951", "Z9481"[:3] + "5"]),
])
def test_icd_rules_spot_checks(name, version, pos, neg):
    assert all(CM.code_matches(c, version, name) for c in pos), [c for c in pos if not CM.code_matches(c, version, name)]
    assert not any(CM.code_matches(c, version, name) for c in neg), [c for c in neg if CM.code_matches(c, version, name)]


def test_icd_versions_are_never_mixed():
    assert not CM.code_matches("250", 10, "diabetes")                 # ICD-9 code string under an ICD-10 rule
    assert not CM.code_matches("E119", 9, "diabetes")                 # ICD-10 code string under an ICD-9 rule
    assert not CM.code_matches("I50", 9, "heart_failure")
    assert not CM.code_matches("428", 10, "heart_failure")
    assert not CM.code_matches("E119", 99, "diabetes") and not CM.code_matches("E119", None, "diabetes")
    assert CM.code_matches("e11.9", 10, "diabetes")                   # dots / lower case are normalised


def test_no_sepsis_or_infection_code_can_leak_into_comorbidities():
    sepsis9 = ["038", "0389", "03811", "99591", "99592", "78552", "7907", "5990", "486"]
    sepsis10 = ["A419", "A4151", "A40", "R652", "R6520", "R6521", "J189", "N390", "N179", "K7200"]
    for name in CM.COMORBIDITIES:
        assert not any(CM.code_matches(c, 9, name) for c in sepsis9), name
        assert not any(CM.code_matches(c, 10, name) for c in sepsis10), name


def test_join_uses_subject_and_hadm_never_subject_alone():
    # patient 1 has two admissions: diabetes only in admission 100. Patient 2 is unrelated.
    cohort = pd.DataFrame({"stay_id": [1, 2, 3, 4], "subject_id": [1, 1, 2, 3], "hadm_id": [100, 101, 200, 300]})
    dx = pd.DataFrame([
        dict(subject_id=1, hadm_id=100, icd_version=10, icd_code="E119"),
        dict(subject_id=1, hadm_id=100, icd_version=10, icd_code="I5022"),
        dict(subject_id=1, hadm_id=100, icd_version=10, icd_code="N189"),
        dict(subject_id=1, hadm_id=100, icd_version=10, icd_code="C349"),        # 4 rows for one admission
        dict(subject_id=2, hadm_id=200, icd_version=9, icd_code="5715"),
        dict(subject_id=99, hadm_id=300, icd_version=10, icd_code="E119"),        # hadm 300 belongs to subject 3, not 99
    ])
    out, rep = CM.build_stay_comorbidities(dx, cohort)
    assert len(out) == len(cohort) and out["stay_id"].tolist() == [1, 2, 3, 4]     # no row multiplication
    o = out.set_index("stay_id")
    assert o.loc[1, ["s_diabetes", "s_heart_failure", "s_chronic_kidney_disease", "s_cancer"]].tolist() == [1, 1, 1, 1]
    assert o.loc[2].drop([]).sum() == 0                                           # same patient, other admission -> 0
    assert o.loc[3, "s_chronic_liver_disease"] == 1 and o.loc[3].sum() == 1
    assert o.loc[4].sum() == 0                                                    # mismatching subject_id is NOT joined
    assert rep["diagnosis_rows_dropped_subject_hadm_mismatch"] == 1
    assert rep["stays_without_any_diagnosis_row"] == 2                            # stays 2 and 4
    assert all(str(t) == "int8" for t in out.dtypes.iloc[1:])


def test_two_icu_stays_in_one_admission_get_identical_flags():
    cohort = pd.DataFrame({"stay_id": [1, 2], "subject_id": [5, 5], "hadm_id": [50, 50]})
    dx = pd.DataFrame([dict(subject_id=5, hadm_id=50, icd_version=10, icd_code="C349")])
    out, _ = CM.build_stay_comorbidities(dx, cohort)
    assert out["s_cancer"].tolist() == [1, 1]


def test_empty_diagnoses_gives_all_zero_not_error():
    cohort = pd.DataFrame({"stay_id": [1], "subject_id": [1], "hadm_id": [10]})
    out, rep = CM.build_stay_comorbidities(pd.DataFrame(columns=["subject_id", "hadm_id", "icd_version", "icd_code"]), cohort)
    assert out.shape == (1, 7) and out.iloc[0, 1:].sum() == 0 and rep["stays_without_any_diagnosis_row"] == 1


def test_adding_comorbidities_changes_nothing_else(tmp_path):
    """Same raw batches, finalize with and without comorbidities: every pre-existing column/row must be identical."""
    cfg_on = json.loads(json.dumps(CFG)); cfg_on["hourly"]["n_batches"] = 4
    cfg_off = json.loads(json.dumps(cfg_on)); cfg_off["comorbidities"]["enabled"] = False
    cohort, _ = refine_cohort(raw_cohort(), cfg_on)
    bq = FakeBQ(cohort)
    raw = tmp_path / "raw"; raw.mkdir()
    for k, b in cohort.groupby("batch_id"):
        extract_batch(bq, b, cfg_on).to_parquet(raw / f"batch_{int(k):02d}.parquet", index=False)
    res = {}
    for tag, cfg in (("off", cfg_off), ("on", cfg_on)):
        d = tmp_path / tag
        shutil.copytree(raw, d / cfg["paths"]["work_dir"] / "raw_batches")
        if tag == "on":
            write_comorb(d, cfg, cohort)
        assert finalize(cfg, d)["validation_passed"]
        res[tag] = {sub: pd.concat(pd.read_parquet(p) for p in sorted((d / cfg["paths"][sub]).glob("batch_*.parquet")))
                    for sub in ("treatment_dir", "early_dir")}
    off, on = res["off"]["treatment_dir"], res["on"]["treatment_dir"]
    assert len(on) == len(off) and list(on["stay_id"]) == list(off["stay_id"]) and list(on["step"]) == list(off["step"])
    assert list(on.columns[:len(off.columns)]) == list(off.columns)               # old columns first, same order
    assert set(on.columns) - set(off.columns) == set(CM.STATE_COMORBIDITY_COLS + CM.NEXT_COMORBIDITY_COLS)
    pd.testing.assert_frame_equal(on[list(off.columns)].reset_index(drop=True), off.reset_index(drop=True))
    for c in ("action_fluid_bin", "action_vaso_bin", "action", "reward", "done", "split"):
        assert on[c].equals(off[c])
    pd.testing.assert_frame_equal(res["on"]["early_dir"].reset_index(drop=True), res["off"]["early_dir"].reset_index(drop=True))
    assert not any(c in res["on"]["early_dir"].columns for c in CM.STATE_COMORBIDITY_COLS)   # early-prediction files untouched


def test_finalize_refuses_to_run_without_comorbidity_table(tmp_path):
    cfg = json.loads(json.dumps(CFG)); cfg["hourly"]["n_batches"] = 2
    cohort, _ = refine_cohort(raw_cohort(n_sep=10, n_ctl=20), cfg)
    bq = FakeBQ(cohort)
    bdir = tmp_path / cfg["paths"]["work_dir"] / "raw_batches"; bdir.mkdir(parents=True)
    for k, b in cohort.groupby("batch_id"):
        extract_batch(bq, b, cfg).to_parquet(bdir / f"batch_{int(k):02d}.parquet", index=False)
    with pytest.raises(RuntimeError, match="run.py comorbidities"):
        finalize(cfg, tmp_path)
