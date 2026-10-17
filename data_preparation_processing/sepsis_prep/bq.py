"""Thin BigQuery wrapper: parameterised queries -> pandas, retries, dry-run, cost log."""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import pandas as pd

log = logging.getLogger("sepsis_prep")


class BQ:
    def __init__(self, cfg: dict[str, Any]):
        b = cfg["bigquery"]
        self.project = b["billing_project"]
        self.location = b.get("location", "US")
        self.max_bytes = int(float(b.get("max_gb_per_query", 50)) * 1e9)
        sp = b.get("source_project", "physionet-data")
        self.HOSP = f"{sp}.{b['hosp_dataset']}"
        self.ICU = f"{sp}.{b['icu_dataset']}"
        self.DERIVED = f"{sp}.{b['derived_dataset']}"
        self._client = None
        self.log: list[tuple[str, float]] = []   # (label, GB billed or estimated)

    def client(self):
        if self._client is None:
            from google.cloud import bigquery
            self._client = bigquery.Client(project=self.project, location=self.location)
        return self._client

    @staticmethod
    def _params(params: dict[str, Any]):
        from google.cloud import bigquery
        out = []
        for k, v in params.items():
            if isinstance(v, (list, tuple)):
                out.append(bigquery.ArrayQueryParameter(k, "INT64", [int(x) for x in v]))
            elif isinstance(v, (datetime, pd.Timestamp)):
                out.append(bigquery.ScalarQueryParameter(k, "DATETIME", pd.Timestamp(v).to_pydatetime()))
            elif isinstance(v, bool):
                out.append(bigquery.ScalarQueryParameter(k, "BOOL", v))
            elif isinstance(v, int):
                out.append(bigquery.ScalarQueryParameter(k, "INT64", v))
            else:
                out.append(bigquery.ScalarQueryParameter(k, "FLOAT64", float(v)))
        return out

    def query_df(self, sql: str, params: dict[str, Any] | None = None, label: str = "query",
                 dry_run: bool = False, tries: int = 4):
        """dry_run=True -> returns estimated GB (free, also validates SQL + column names)."""
        from google.cloud import bigquery
        from google.api_core import exceptions as gex

        cfg = bigquery.QueryJobConfig(
            query_parameters=self._params(params or {}),
            dry_run=dry_run, use_query_cache=not dry_run,
            maximum_bytes_billed=self.max_bytes,
        )
        for attempt in range(1, tries + 1):
            try:
                job = self.client().query(sql, job_config=cfg)
                if dry_run:
                    gb = float(job.total_bytes_processed or 0) / 1e9
                    self.log.append((label, gb))
                    return gb
                tbl = job.result().to_arrow(create_bqstorage_client=False)
                gb = float(getattr(job, "total_bytes_billed", 0) or 0) / 1e9
                self.log.append((label, gb))
                df = tbl.to_pandas()
                log.info("%-22s %8d rows  %6.2f GB billed", label, len(df), gb)
                return df
            except (gex.ServiceUnavailable, gex.InternalServerError, gex.TooManyRequests,
                    gex.DeadlineExceeded, ConnectionError, TimeoutError) as e:
                wait = 5 * attempt
                log.warning("%s: transient error (%s). retry %d/%d in %ds", label, type(e).__name__, attempt, tries, wait)
                time.sleep(wait)
        raise RuntimeError(f"{label}: failed after {tries} tries (network / BigQuery down?)")

    def total_gb(self) -> float:
        return sum(g for _, g in self.log)
