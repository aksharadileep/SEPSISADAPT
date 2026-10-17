"""All SQL in one file. Every query is filtered server-side by stay_id / hadm_id and a time window,
and aggregated to ONE ROW PER (stay, hour) inside BigQuery, so very little data comes to the laptop.

Time convention: hour_end = end of the hour bucket = first moment the information is available."""
from __future__ import annotations

from .features import LAB_TABLES, VITALS

HOUR_END = "DATETIME_ADD(DATETIME_TRUNC({t}, HOUR), INTERVAL 1 HOUR)"


def cohort_sql(D, min_age=True):
    return f"""
WITH icu AS (
  SELECT i.subject_id, i.hadm_id, i.stay_id, i.first_careunit, i.intime, i.outtime,
         DATETIME_DIFF(i.outtime, i.intime, MINUTE) / 60.0 AS los_hours,
         ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime) AS stay_rank
  FROM `{D.ICU}.icustays` i
),
sep AS (
  SELECT stay_id, suspected_infection_time, sofa_time
  FROM `{D.DERIVED}.sepsis3`
  WHERE sepsis3 IS TRUE
)
SELECT icu.subject_id, icu.hadm_id, icu.stay_id, icu.first_careunit, icu.intime, icu.outtime,
       icu.los_hours, icu.stay_rank,
       p.gender, p.anchor_age AS age,
       a.hospital_expire_flag,
       sep.suspected_infection_time, sep.sofa_time,
       (sep.stay_id IS NOT NULL) AS is_sepsis
FROM icu
JOIN `{D.HOSP}.patients`   p USING (subject_id)
JOIN `{D.HOSP}.admissions` a USING (hadm_id)
LEFT JOIN sep USING (stay_id)
WHERE p.anchor_age >= @min_age
"""


def vitals_sql(D):
    cols = ",\n       ".join(f"AVG({c}) AS {c}" for c in VITALS)
    return f"""
SELECT stay_id, {HOUR_END.format(t='charttime')} AS hour_end,
       {cols}
FROM `{D.DERIVED}.vitalsign`
WHERE stay_id IN UNNEST(@stay_ids) AND charttime >= @t_start AND charttime < @t_end
GROUP BY stay_id, hour_end
"""


def gcs_sql(D):
    return f"""
SELECT stay_id, {HOUR_END.format(t='charttime')} AS hour_end, MIN(gcs) AS gcs
FROM `{D.DERIVED}.gcs`
WHERE stay_id IN UNNEST(@stay_ids) AND charttime >= @t_start AND charttime < @t_end
GROUP BY stay_id, hour_end
"""


def lab_sql(D, table):
    cols = ",\n       ".join(f"AVG({c}) AS {c}" for c in LAB_TABLES[table])
    return f"""
SELECT hadm_id, {HOUR_END.format(t='charttime')} AS hour_end,
       {cols}
FROM `{D.DERIVED}.{table}`
WHERE hadm_id IN UNNEST(@hadm_ids) AND charttime >= @lab_start AND charttime < @t_end
GROUP BY hadm_id, hour_end
"""


def urine_sql(D):
    return f"""
SELECT stay_id, {HOUR_END.format(t='charttime')} AS hour_end, SUM(urineoutput) AS urine_ml
FROM `{D.DERIVED}.urine_output`
WHERE stay_id IN UNNEST(@stay_ids) AND charttime >= @t_start AND charttime < @t_end
GROUP BY stay_id, hour_end
"""


def inputevents_sql(D):
    # fluids + vasopressors in one pass (interval records, few rows)
    return f"""
SELECT stay_id, starttime, endtime, itemid, amount, amountuom, rate, rateuom, patientweight
FROM `{D.ICU}.inputevents`
WHERE stay_id IN UNNEST(@stay_ids) AND itemid IN UNNEST(@item_ids)
  AND starttime < @t_end AND endtime >= @t_start
"""


def ventilation_sql(D):
    return f"""
SELECT stay_id, starttime, endtime, ventilation_status
FROM `{D.DERIVED}.ventilation`
WHERE stay_id IN UNNEST(@stay_ids) AND starttime < @t_end AND endtime >= @t_start
"""


def sofa_sql(D):
    return f"""
SELECT stay_id, endtime AS hour_end, sofa_24hours AS sofa_24h
FROM `{D.DERIVED}.sofa`
WHERE stay_id IN UNNEST(@stay_ids) AND endtime >= @t_start AND endtime < @t_end
"""


def diagnoses_sql(D):
    """Comorbidity source. Both keys are filtered; the Python side joins on (subject_id, hadm_id) again."""
    return f"""
SELECT subject_id, hadm_id, icd_version, icd_code
FROM `{D.HOSP}.diagnoses_icd`
WHERE hadm_id IN UNNEST(@hadm_ids) AND subject_id IN UNNEST(@subject_ids)
"""


def d_items_sql(D):
    return f"""
SELECT itemid, label, category, unitname
FROM `{D.ICU}.d_items`
WHERE itemid IN UNNEST(@item_ids)
ORDER BY itemid
"""


# label -> (builder, key columns needed)
def all_batch_queries(D):
    q = {
        "vitals": vitals_sql(D),
        "gcs": gcs_sql(D),
        "urine": urine_sql(D),
        "inputevents": inputevents_sql(D),
        "ventilation": ventilation_sql(D),
        "sofa": sofa_sql(D),
    }
    for t in LAB_TABLES:
        q[t] = lab_sql(D, t)
    return q
