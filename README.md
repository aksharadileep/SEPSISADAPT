# Sepsis data preparation (MIMIC-IV on BigQuery) — batch-wise, 8 GB laptop friendly

Builds the datasets for your project from **both sepsis and non-sepsis ICU patients**:

1. **Early-prediction data** – one row per patient per hour, label = *"sepsis starts within the next 6 h"*.
2. **Treatment data** – for sepsis patients only: state / action (IV fluid × vasopressor, 25 actions) / reward / next-state, one row per 4 h.

Nothing is trained here. No MIMIC CSV files needed – data is read from BigQuery, **in exactly 32 batches**
(`hourly.n_batches` in `config.yaml`), and each finished batch is saved. Every batch contains **both sepsis and non-sepsis patients**
(sepsis and non-sepsis stays are shuffled separately and dealt out evenly), and all 32 batches have the same size. If the internet or laptop dies, run the same command again — it continues.

## Run it (copy-paste)

```bash
# 1. once: environment
python -m venv .venv
.venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 2. once: Google login (same Google account that has MIMIC-IV access on PhysioNet/BigQuery)
gcloud auth application-default login
gcloud auth application-default set-quota-project my-sepsis-project-123456

# 3. check everything (FREE, nothing is billed)
python run.py check
python run.py audit-items          # look at the fluid labels, fix fluid_itemids in config.yaml if any is wrong

# 4. tiny test first (200 stays, a few minutes)
python run.py smoke

# 5. the real run
python run.py cohort
python run.py estimate             # free: shows how many GB BigQuery will read
python run.py extract              # long step, resumable. Run again if it stops.
python run.py finalize
# (or just:  python run.py all)
python run.py status               # progress any time
```
If your GCP project id is not `my-sepsis-project-123456`, change `billing_project` in `config.yaml`.

## What you get

```
output/
├── early_prediction/            batch_01.parquet ... batch_32.parquet   sepsis + non-sepsis patients, 1 row = 1 patient-hour
├── treatment_recommendation/    batch_01.parquet ... batch_32.parquet   sepsis patients only, 1 row = 1 four-hour treatment step
├── feature_columns.json         which columns are model inputs / labels / must never be inputs
├── train_stats.json             train-only medians, mean, std + the fluid & vasopressor bin cut-points
├── data_report.json             counts per split AND per batch, label prevalence, validation result
└── _work/                       cohort + raw batches (needed to resume; you may delete it when everything is finished)
```
Every row has a `split` column (train / validation / test, decided per **patient**), so you can still split without mixing patients.

* `early_prediction` label: `label_sepsis_6h` = 1 if sepsis starts within the next 6 h (also `_12h`, `_24h`). `is_sepsis_stay` tells sepsis vs non-sepsis patient.
* `treatment_recommendation` columns: patient/stay ids, `s_*` = patient state before the step (vitals, labs, age, sex, weight, SOFA ...),
  `fluid_block` (mL given) and `vaso_block` (norepinephrine-equivalent) = **the treatment actually given**, `action_fluid_bin`, `action_vaso_bin`, `action` (0..24),
  `reward`, `done`, `ns_*` = state after the step, `hospital_expire_flag` = outcome.

Load one batch at a time (8 GB RAM friendly):
```python
import pandas as pd, json
fc = json.load(open("output/feature_columns.json"))
b = pd.read_parquet("output/early_prediction/batch_01.parquet")
X, y = b[fc["model_inputs"]], b["label_sepsis_6h"]
steps = pd.read_parquet("output/treatment_recommendation/batch_01.parquet")
```
Training in a loop over the 32 files (never loads everything at once):
```python
from pathlib import Path
for f in sorted(Path("output/early_prediction").glob("batch_*.parquet")):
    b = pd.read_parquet(f); tr = b[b["split"] == "train"]
    # model.partial_fit(tr[fc["model_inputs"]], tr["label_sepsis_6h"])   # or one training epoch per file
```
Want CSV instead? set `paths.file_format: "csv"` in `config.yaml` and run `python run.py finalize` again.

## Rules built in (so the data is honest)
* **Patient-level split** 70/15/15 – one patient is never in two splits.
* **Label**: 1 only if sepsis onset is *after* the row and within 6 h (also 12 h, 24 h columns). Rows at/after onset are **not** in the prediction files.
* **No future leakage**: row `timestamp` = end of the hour; values carried forward only (vitals ≤ 6 h, labs ≤ 24 h), never backward.
* **Missing values**: filled with **train-only** medians; `<feature>_obs` = 1 means really measured that hour.
* Garbage values (HR 999 etc.) become missing (`plausible_ranges` in config).
* **Sepsis** = MIMIC `derived.sepsis3` with `sepsis3 = TRUE`. **Non-sepsis** = every other adult ICU stay (includes patients with infection but no sepsis — the harder, realistic controls).
* Sepsis time-zero rule is `latest` = first moment infection **and** organ dysfunction both hold (change `onset_rule` in config).
* `sofa_24h` is provided but **not** in `model_inputs` (the Sepsis-3 label is defined with SOFA, so it makes the task artificially easy). Add it from `optional_inputs` if you want.
* Treatment actions: fluid mL and vasopressor (norepinephrine-equivalent) per 4 h → bins 0 (none) + 4 quartiles of non-zero doses from TRAIN → 5×5 = 25 actions. Reward = SOFA improvement, ±15 at the end for survival.

## Comorbidities (treatment table, RL context features)
Six static 0/1 columns, added to `treatment_recommendation` as **state context** (never actions):
`s_diabetes`, `s_heart_failure`, `s_chronic_kidney_disease`, `s_chronic_liver_disease`, `s_cancer`, `s_immunosuppression`
(plus identical `ns_*` twins so state and next_state have the same columns; set `comorbidities.add_next_state_columns: false` to drop them).

```bash
python run.py comorbidities   # after `cohort`; one small query on diagnoses_icd, writes _work/comorbidities.parquet
python run.py finalize        # merges them; refuses to run if the file is missing
```
* Source `hosp.diagnoses_icd`, ICD-9 **and** ICD-10 (matched per version). Rules + clinical scope: `sepsis_prep/comorbidity.py`.
* Link: `(subject_id, hadm_id)` -> `stay_id`. Never `subject_id` alone. Diagnosis rows whose pair does not match the cohort are dropped and counted.
* A stay whose admission has no diagnosis rows gets 0 ("not documented"); the count is in `_work/comorbidity_report.json`.
* Merged **after** action / reward / done are built; the code asserts that every pre-existing column and row is unchanged.
* No sepsis, infection or acute-organ-failure code is used by any rule (unit-tested).
* **Limitation (important):** `diagnoses_icd` is coded at hospital *discharge* and has no timestamp. The six conditions are chronic,
  so this is the standard practice, but a few codes (e.g. a cancer first diagnosed during the stay) can be documented only
  at discharge. Stricter alternative if you need it: use only codes from the patient's *earlier* admissions.
* Existing 32 raw batches do **not** need to be re-extracted; `batch_id` and `split` are unchanged.

## Size / time knobs (`config.yaml`)
* RAM problem → raise `hourly.n_batches` (e.g. 64, smaller batches). Cheaper BigQuery bill → fewer batches.
* Smaller dataset → `cohort.control_ratio: 3` (3 non-sepsis stays per sepsis stay) or `cohort.limit_stays`.
* Changing cohort settings after `extract` started → delete `output/_work/` and run `cohort` + `extract` again (it will tell you).

## If something fails
* `check` says *Access Denied* → redo step 2, and confirm your MIMIC-IV BigQuery access.
* `check` says *Unrecognized name* → that derived table has a different column in your dataset; send me the line.
* Stopped midway → run the same command again. Finished batches are never redone.
* Everything is logged in `output/_work/run.log`.

Tests (no internet needed): `pip install pytest && python -m pytest -q`
