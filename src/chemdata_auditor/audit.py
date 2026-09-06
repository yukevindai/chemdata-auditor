"""Configurable checks; findings are evidence for review, never automatic repairs."""

from dataclasses import dataclass, field

import numpy as np

from .common import JsonResult, finite_number, frame, metadata, missing, names, numeric, require


@dataclass
class AuditConfig:
    duplicate_columns: list[str] | None = None
    group_columns: list[str] = field(default_factory=list)
    split_column: str | None = None
    bounds: dict[str, list[float | None]] = field(default_factory=dict)
    units: dict[str, dict[str, str]] = field(default_factory=dict)
    provenance_columns: list[str] = field(default_factory=list)
    sparse_bins: dict[str, list[float]] = field(default_factory=dict)
    min_bin_count: int = 3
    feature_columns: list[str] = field(default_factory=list)
    unavailable_features: list[str] = field(default_factory=list)
    target_column: str | None = None

    def __post_init__(self):
        for key in ("group_columns", "provenance_columns", "feature_columns", "unavailable_features"):
            names(getattr(self, key), key)
        if self.duplicate_columns is not None:
            names(self.duplicate_columns, "duplicate_columns")
            if not self.duplicate_columns:
                raise ValueError("duplicate_columns cannot be empty.")
        for key in ("split_column", "target_column"):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{key} must be a nonempty column name.")
        if self.split_column and self.duplicate_columns and self.split_column in self.duplicate_columns:
            raise ValueError("duplicate_columns must not include split_column; it would hide cross-partition duplicates.")
        if self.split_column in self.group_columns:
            raise ValueError("group_columns must not include split_column.")
        for key in ("bounds", "units", "sparse_bins"):
            value = getattr(self, key)
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object.")
            names(list(value), key)
        if isinstance(self.min_bin_count, bool) or not isinstance(self.min_bin_count, int) or self.min_bin_count < 1:
            raise ValueError("min_bin_count must be a positive integer.")
        for column, limits in self.bounds.items():
            if not isinstance(limits, (list, tuple)) or len(limits) != 2 or all(x is None for x in limits):
                raise ValueError(f"Bounds for {column!r} need [lower, upper], at least one finite.")
            for value in limits:
                if value is not None:
                    finite_number(value, f"Bound for {column}")
            if None not in limits and limits[0] > limits[1]:
                raise ValueError(f"Reversed bounds for {column!r}.")
        for column, rule in self.units.items():
            if not isinstance(rule, dict) or set(rule) != {"column", "expected"}:
                raise ValueError(f"Unit rule for {column!r} needs column and expected only.")
            if any(not isinstance(x, str) or not x.strip() for x in rule.values()):
                raise ValueError("Unit column and expected unit must be nonempty strings.")
        for column, edges in self.sparse_bins.items():
            if not isinstance(edges, (list, tuple)) or len(edges) < 2:
                raise ValueError(f"Sparse bins for {column!r} need at least two edges.")
            for edge in edges:
                finite_number(edge, f"Bin edge for {column}")
            if np.any(np.diff(edges) <= 0):
                raise ValueError("Sparse bin edges must be strictly increasing.")


@dataclass
class Finding:
    code: str
    severity: str
    columns: list[str]
    rows: list[int]
    message: str
    suggestion: str
    details: dict = field(default_factory=dict)


@dataclass
class AuditReport(JsonResult):
    n_rows: int
    findings: list[Finding]
    checks_run: list[str]
    checks_skipped: dict[str, str]
    metadata: dict


def audit(data, config=None):
    """Return a JSON-serializable report without modifying the input DataFrame."""
    cfg = config or AuditConfig()
    data = frame(data)
    duplicates = cfg.duplicate_columns or [c for c in data if c != cfg.split_column]
    if not duplicates:
        raise ValueError("No columns remain for duplicate comparison.")
    required = (duplicates + cfg.group_columns + list(cfg.bounds) + list(cfg.units)
                + list(cfg.sparse_bins) + cfg.feature_columns)
    require(data, required)
    if cfg.split_column:
        require(data, [cfg.split_column])
    if cfg.target_column:
        require(data, [cfg.target_column])
    findings, ran, skipped = [], [], {}

    def add(code, severity, columns, mask, message, suggestion, details=None):
        rows = np.flatnonzero(np.asarray(mask, dtype=bool)).tolist()
        findings.append(Finding(code, severity, columns, rows, message, suggestion, details or {}))

    ran.append("duplicates")
    duplicated = data.duplicated(subset=duplicates, keep=False)
    if duplicated.any():
        add("duplicate_samples", "warning", duplicates, duplicated,
            "Records repeat on the comparison columns; missing values compare equal.",
            "Check whether these are duplicate records or legitimate replicate measurements.")

    if cfg.split_column:
        ran.append("split_leakage")
        label = cfg.split_column
        absent = missing(data[label])
        if absent.any():
            add("missing_split", "error", [label], absent, "Some rows have no split assignment.",
                "Assign or explicitly exclude these rows before evaluation.")
        valid = data.loc[~absent]
        # Compare every label pair, including validation; all labels are partitions.
        overlap = valid.groupby(duplicates, dropna=False)[label].transform("nunique") > 1
        if overlap.any():
            add("duplicate_split_overlap", "error", duplicates + [label], overlap.reindex(data.index, fill_value=False),
                "Matching records occur in multiple partitions.", "Keep duplicate records in one partition.")
        for col in cfg.group_columns:
            absent_group = missing(data[col])
            if absent_group.any():
                add("missing_group", "warning", [col], absent_group,
                    "Missing group identifiers prevent a complete overlap check.", "Recover the sample or group metadata.")
            known = data.loc[~absent & ~absent_group]
            overlap = known.groupby(col, dropna=False)[label].transform("nunique") > 1
            if overlap.any():
                add("group_split_overlap", "error", [col, label], overlap.reindex(data.index, fill_value=False),
                    "A declared group occurs in multiple partitions.", "Keep the entire group in one partition.")
    else:
        skipped["split_leakage"] = "No split_column configured."

    if cfg.feature_columns:
        ran.append("feature_leakage")
        for col in cfg.feature_columns:
            if col in cfg.unavailable_features or col == cfg.target_column:
                add("unavailable_feature", "error", [col], np.ones(len(data), bool),
                    "A model feature is the target or declared unavailable at prediction time.", "Remove it from model inputs.")
            elif cfg.target_column:
                usable = ~missing(data[col]) & ~missing(data[cfg.target_column])
                if usable.sum() >= 3 and data.loc[usable, col].eq(data.loc[usable, cfg.target_column]).all():
                    add("target_copy", "warning", [col, cfg.target_column], usable,
                        "A feature exactly matches the target on all jointly observed rows (at least three).",
                        "Investigate whether the feature copies or encodes the outcome.")
    else:
        skipped["feature_leakage"] = "No feature_columns configured."

    if cfg.bounds or cfg.sparse_bins:
        ran.append("numeric_validity")
    values = {}
    for col in dict.fromkeys([*cfg.bounds, *cfg.sparse_bins]):
        values[col] = numeric(data[col])
        invalid = ~np.isfinite(values[col])
        if invalid.any():
            add("invalid_numeric", "error", [col], invalid,
                "Missing, nonnumeric, or nonfinite measurement.", "Recover or explicitly exclude invalid measurements.")

    if cfg.bounds:
        ran.append("bounds")
        for col, (lower, upper) in cfg.bounds.items():
            x = values[col]
            bad = np.isfinite(x) & ((x < lower if lower is not None else False) | (x > upper if upper is not None else False))
            if bad.any():
                add("out_of_bounds", "error", [col], bad, "Values violate inclusive configured bounds.",
                    "Verify the measurement, unit, and scientific applicability of the bounds.", {"lower": lower, "upper": upper})
    else:
        skipped["bounds"] = "No physical bounds configured."

    if cfg.units:
        ran.append("units")
        for col, rule in cfg.units.items():
            unit_col, expected = rule["column"], rule["expected"].strip()
            if unit_col not in data:
                add("missing_unit_column", "error", [col, unit_col], np.ones(len(data), bool),
                    "The configured unit metadata column is absent.", "Supply explicit unit metadata.")
                continue
            absent = missing(data[unit_col])
            mismatch = ~absent & data[unit_col].astype(str).str.strip().ne(expected)
            if absent.any():
                add("missing_unit", "error", [col, unit_col], absent, "Unit metadata is missing.", "Recover units from the source.")
            if mismatch.any():
                add("unit_mismatch", "warning", [col, unit_col], mismatch,
                    f"Unit labels differ from the configured canonical label {expected!r}.",
                    "Check compatibility and convert explicitly; this check does not convert or infer units.", {"expected": expected})
    else:
        skipped["units"] = "No unit rules configured."

    if cfg.provenance_columns:
        ran.append("provenance")
        for col in cfg.provenance_columns:
            absent = missing(data[col]) if col in data else np.ones(len(data), bool)
            if np.any(absent):
                add("provenance_gap", "warning", [col], absent, "Required provenance metadata is missing or blank.",
                    "Recover source references and experimental context; do not fabricate missing metadata.")
    else:
        skipped["provenance"] = "No provenance_columns configured."

    if cfg.sparse_bins:
        ran.append("sparse_regions")
        for col, edges in cfg.sparse_bins.items():
            x = values[col]
            for i, (left, right) in enumerate(zip(edges, edges[1:])):
                last = i == len(edges) - 2
                mask = np.isfinite(x) & (x >= left) & ((x <= right) if last else (x < right))
                count = int(mask.sum())
                if count < cfg.min_bin_count:
                    add("sparse_bin", "info", [col], mask, "A configured one-dimensional interval has few observations.",
                        "Inspect coverage; counts describe rows, not independent experiments.",
                        {"left": left, "right": right, "right_inclusive": last, "count": count})
            outside = np.isfinite(x) & ((x < edges[0]) | (x > edges[-1]))
            if outside.any():
                add("outside_sparse_bins", "info", [col], outside, "Rows fall outside the configured coverage intervals.",
                    "Expand the bins if these rows are within the intended study domain.")
    else:
        skipped["sparse_regions"] = "No sparse_bins configured."
    return AuditReport(len(data), findings, ran, skipped, metadata(data, cfg))
