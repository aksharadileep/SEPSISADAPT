# Data Preparation and Processing

**Module 1 of SEPSISADAPT.** This pipeline reads MIMIC-IV v3.1 directly from Google BigQuery and produces the two datasets used by the rest of the system:

| Dataset | Granularity | Used by |
|---------|-------------|---------|
| Early-prediction dataset | One row per patient-hour (sepsis and non-sepsis patients) | Module 2: Early sepsis prediction (Temporal Transformer) |
| Treatment dataset | One row per 4-hour step (sepsis patients only) | Module 3: Treatment recommendation (Conservative Q-Learning) |

Both datasets are built from **one common cohort** and split **by patient** into train, validation and test sets.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Folder Structure](#folder-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the Pipeline](#running-the-pipeline)
7. [Processing Steps](#processing-steps)
8. [Output Datasets](#output-datasets)
9. [Design Decisions](#design-decisions)
10. [Testing](#testing)
11. [Troubleshooting](#troubleshooting)

---

## Pipeline Overview

```
MIMIC-IV v3.1 (BigQuery)
        |
        v
 Step 1  Base cohort (ICU stays)
        |
        v
 Step 2  Demographics (static variables)
        |
        v
 Step 3  Comorbidity flags (6 conditions)
        |
        v
 Step 4  Hourly vital signs
        |
        v
 Step 5  Hourly laboratory values + forward-fill
        |
        v
 Step 6  Labels, observation flags and treatments
        |
        +-----------------------------+
        v                             v
 Early-prediction dataset      Step 7  Treatment dataset
 (patient-hour rows)           (state, action, reward, next state)
        |                             |
        +--------------+--------------+
                       v
 Step 8  Finalize: patient-level train / validation / test split
```

---

## Folder Structure

```
data_preparation_processing/
|-- README.md             This file
|-- config.yaml           Pipeline configuration
|-- requirements.txt      Python dependencies
|-- run.py                Pipeline entry point
|-- sepsis_prep/          Pipeline package
|   |-- bq.py             BigQuery client and query helpers
|   |-- sql.py            SQL query definitions
|   |-- cohort.py         Step 1: base cohort
|   |-- comorbidity.py    Step 3: comorbidity flags
|   |-- hourly.py         Steps 4-5: hourly vitals and labs
|   |-- features.py       Feature definitions
|   |-- treatment.py      Step 7: treatment (state, action, reward) table
|   |-- finalize.py       Step 8: final datasets and splits
|   `-- __init__.py
`-- tests/
    `-- test_pipeline.py  Pipeline tests
```

---

## Prerequisites

- Python 3.9 or later
- A Google Cloud project with BigQuery enabled
- Credentialed access to MIMIC-IV v3.1 on BigQuery (PhysioNet account, CITI training and signed data use agreement)
- Authenticated Google Cloud credentials on your machine

> **Data policy:** MIMIC-IV is a credentialed dataset. Never commit patient data, exported tables or credential files to this repository.

---

## Installation

```bash
git clone https://github.com/aksharadileep/SEPSISADAPT.git
cd SEPSISADAPT/data_preparation_processing

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

Authenticate with Google Cloud:

```bash
gcloud auth application-default login
```

---

## Configuration

All settings are in `config.yaml`. Review it before the first run, in particular:

- the Google Cloud project used to run the queries
- the MIMIC-IV dataset location on BigQuery
- the output directory for the generated datasets
- the list of measured variables, the prediction horizons and the treatment step length

Keep project identifiers and paths in the config file. Do not hard-code them in the source files.

---

## Running the Pipeline

From inside the `data_preparation_processing` folder:

```bash
python run.py
```

Run it from this folder so that relative paths in `config.yaml` resolve correctly.

---

## Processing Steps

### Step 1: Base cohort

`sepsis_prep/cohort.py`

The cohort is built from the ICU stays table (`icustays`). Each ICU stay is one record in the cohort and carries the identifiers (patient, hospital admission, ICU stay) that every later step joins on. This single cohort feeds both output datasets, so the early-prediction data and the treatment data always describe the same patients.

### Step 2: Demographics (static variables)

Static patient information is attached to each cohort record. These variables do not change over the stay and are repeated on every hourly row in the final dataset.

### Step 3: Comorbidity flags

`sepsis_prep/comorbidity.py`

Pre-existing conditions are derived from the hospital diagnosis codes (`diagnoses_icd`). Six binary flags are produced for each patient:

| Flag | Condition |
|------|-----------|
| 1 | Diabetes |
| 2 | Chronic kidney disease (CKD) |
| 3 | Congestive heart failure (CHF) |
| 4 | Liver disease |
| 5 | Cancer |
| 6 | Immunosuppression |

The flags are used as explicit state variables in the treatment dataset (Module 3).

### Step 4: Hourly vital signs

`sepsis_prep/hourly.py`, `sepsis_prep/features.py`

Vital signs are extracted from the charted events table (`chartevents`) and aggregated onto an hourly grid that starts at ICU admission. Each ICU stay becomes a sequence of one-hour rows.

### Step 5: Hourly laboratory values and forward-fill

`sepsis_prep/hourly.py`

Laboratory results are extracted from the lab events table (`labevents`) and placed on the same hourly grid. Labs are measured irregularly, so most hours have no new value. Missing hours are filled by carrying the last measured value forward in time. Values are only ever carried forward, never backward, so no future information leaks into earlier hours.

### Step 6: Labels, observation flags and treatments

`sepsis_prep/hourly.py`, `sepsis_prep/finalize.py`

- **Observation flags.** For each of the 33 measured variables, a binary flag records whether the value in that hour was really measured (1) or carried forward or filled (0). The model can use these flags to learn how much to trust each value.
- **Prediction labels.** For every patient-hour, labels mark whether sepsis begins within the next 6, 12 and 24 hours. The 6-hour label (`label_sepsis_6h`) is the primary prediction target.
- **Treatments.** IV fluid and vasopressor information is added to the hourly records.

### Step 7: Treatment dataset (state, action, reward)

`sepsis_prep/treatment.py`

For sepsis patients only, the hourly data around sepsis onset is converted into a reinforcement-learning dataset with one row per **4-hour step**:

| Column | Description |
|--------|-------------|
| state | 41 patient variables at the start of the step |
| comorbidity flags | The six flags from Step 3, appended to the state |
| action | One of 25 discrete actions: 5 IV fluid groups x 5 vasopressor groups |
| reward | Improvement in SOFA score over the step |
| next state | State at the start of the following step |
| done flag | Marks the final step of an episode |

A terminal reward of **+15** is added for survival or **-15** for death at the end of the episode.

### Step 8: Finalize and split

`sepsis_prep/finalize.py`

The final tables are written and divided into **train, validation and test** sets. The split is made **by patient**, so all hours and steps of one patient fall in exactly one set. This prevents leakage between sets.

---

## Output Datasets

### Early-prediction dataset

| Content | Count / description |
|---------|---------------------|
| Measured variables | 33 |
| Observation flags | 33 (one per measured variable) |
| Treatments | IV fluid and vasopressor information |
| Static variables | Demographics and related fields |
| Labels | Sepsis onset within 6, 12 and 24 hours |
| Granularity | One row per patient-hour |

### Treatment dataset

| Content | Count / description |
|---------|---------------------|
| State | 41 variables plus 6 comorbidity flags |
| Actions | 25 (5 fluid groups x 5 vasopressor groups) |
| Reward | SOFA improvement, plus +15 / -15 terminal reward |
| Granularity | One row per 4-hour step |
| Population | Sepsis patients only |

---

## Design Decisions

| Decision | Reason |
|----------|--------|
| One common cohort for both datasets | Keeps the prediction and treatment data consistent |
| Direct reading from BigQuery | No need to download the large MIMIC-IV tables locally |
| Forward-fill with observation flags | Handles irregular measurements without hiding which values are real |
| Forward-only filling | Prevents information from the future leaking into earlier hours |
| Patient-level split | Prevents leakage between train, validation and test sets |
| Treatment data from sepsis patients only | Treatment learning needs the data around sepsis onset |
| 4-hour steps | Standard step length for sepsis reinforcement-learning setups |
| Comorbidity flags in the RL state | Lets the recommendation depend on the patient's pre-existing conditions |

---

## Testing

```bash
pip install pytest
pytest tests/
```

The tests in `tests/test_pipeline.py` check the pipeline logic.

---

|
