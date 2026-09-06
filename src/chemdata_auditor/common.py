"""Validation and reproducibility helpers shared by both tools."""

import hashlib
import json
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd


def missing(series):
    return series.isna() | series.map(lambda x: isinstance(x, str) and not x.strip())


def frame(data):
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Supply a nonempty pandas DataFrame.")
    if not data.columns.is_unique or not all(isinstance(c, str) for c in data.columns):
        raise ValueError("Column names must be unique strings.")
    # All public row references are positions, independent of pandas index labels.
    return data.reset_index(drop=True).copy()


def require(data, columns, complete=False):
    absent = [c for c in columns if c not in data]
    if absent:
        raise ValueError(f"Missing required columns: {absent}")
    if complete:
        for c in columns:
            if missing(data[c]).any():
                raise ValueError(f"Column {c!r} has missing or blank values.")


def numeric(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if any(isinstance(x, (bool, np.bool_)) for x in series):
        raise ValueError(f"Boolean values are not numeric measurements in {series.name!r}.")
    return values


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")


def names(value, name):
    if not isinstance(value, (list, tuple)) or any(not isinstance(c, str) or not c for c in value):
        raise ValueError(f"{name} must be a list of nonempty column names.")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicate column names.")


def metadata(data, config):
    versions = {}
    for package in ("numpy", "pandas", "rdkit"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    payload = data.to_json(orient="split", index=False, date_format="iso", double_precision=15)
    return {
        "tool_version": "0.1.0",
        "row_reference": "zero-based row position; use df.iloc, not df.loc",
        "dataset_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "fingerprint_format": "pandas split JSON, index excluded, 15-digit float precision",
        "config": asdict(config),
        "dependencies": versions,
    }


class JsonResult:
    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
