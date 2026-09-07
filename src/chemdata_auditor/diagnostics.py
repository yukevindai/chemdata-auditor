"""Partition overlap, distribution shift, and training-only applicability diagnostics."""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .audit import Finding
from .common import finite_number, frame, metadata, missing, names, numeric, require
from .similarity import matrix, molecules, molecular_pairs, numeric_pairs, pair_budget


@dataclass
class DiagnosticsConfig:
    group_columns: list[str] = field(default_factory=list)
    duplicate_columns: list[str] | None = None
    near_duplicates: dict[str, float] = field(default_factory=dict)
    smiles_column: str | None = None
    molecular_similarity: float | None = None
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    shift_threshold: float = 0.2
    domain_columns: list[str] = field(default_factory=list)
    domain_scales: dict[str, float] = field(default_factory=dict)
    domain_quantile: float = 0.95
    domain_distance_threshold: float | None = None
    max_pair_comparisons: int = 2000000

    def __post_init__(self):
        for key in ("group_columns", "numeric_columns", "categorical_columns", "domain_columns"):
            names(getattr(self, key), key)
        if self.duplicate_columns is not None:
            names(self.duplicate_columns, "duplicate_columns")
            if not self.duplicate_columns:
                raise ValueError("duplicate_columns cannot be empty.")
        for key in ("near_duplicates", "domain_scales"):
            value = getattr(self, key)
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object.")
            names(list(value), key)
            for v in value.values():
                finite_number(v, key)
                if v < 0 or (key == "domain_scales" and v == 0):
                    raise ValueError("Tolerances must be nonnegative and scales positive.")
        if self.domain_scales and set(self.domain_scales) != set(self.domain_columns):
            raise ValueError("Provide a domain scale for every domain column, or use training-only scales.")
        for key in ("shift_threshold", "domain_quantile"):
            finite_number(getattr(self, key), key)
            if not 0 < getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in (0, 1].")
        if self.domain_distance_threshold is not None:
            finite_number(self.domain_distance_threshold, "domain_distance_threshold")
            if self.domain_distance_threshold < 0:
                raise ValueError("domain_distance_threshold must be nonnegative.")
        if isinstance(self.max_pair_comparisons, bool) or not isinstance(self.max_pair_comparisons, int) or self.max_pair_comparisons < 1:
            raise ValueError("max_pair_comparisons must be a positive integer.")
        if self.smiles_column is not None and (not isinstance(self.smiles_column, str) or not self.smiles_column):
            raise ValueError("smiles_column must be a name.")
        if self.molecular_similarity is not None:
            finite_number(self.molecular_similarity, "molecular_similarity")
            if not self.smiles_column or not 0 < self.molecular_similarity <= 1:
                raise ValueError("molecular_similarity needs smiles_column and a threshold in (0,1].")


def validate_partition(result, n):
    parts = {"train": result.train, "validation": result.validation, "test": result.test, "excluded": result.excluded}
    rows = [r for values in parts.values() for r in values]
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in rows):
        raise ValueError("Partition rows must be integer positions.")
    if len(rows) != n or set(rows) != set(range(n)):
        raise ValueError("Partitions must cover each input row exactly once with no overlap.")
    if not result.train or not result.test:
        raise ValueError("Train and test must be nonempty.")
    return {key: value for key, value in parts.items() if value and key != "excluded"}


def diagnose(data, result, config=None):
    """Diagnose an existing result; return JSON-safe metrics plus structured findings."""
    cfg = config or DiagnosticsConfig()
    cfg.__post_init__()
    data = frame(data)
    expected = result.metadata.get("dataset_sha256")
    if expected is not None and expected != metadata(data, cfg)["dataset_sha256"]:
        raise ValueError("Split assignments belong to a different dataset snapshot or row order.")
    parts = validate_partition(result, len(data))
    labels = result.assignments()
    columns = cfg.duplicate_columns or list(data.columns)
    require(data, columns + cfg.group_columns + cfg.numeric_columns + cfg.categorical_columns
            + cfg.domain_columns + list(cfg.near_duplicates) + ([cfg.smiles_column] if cfg.smiles_column else []))
    findings, metrics = [], {"partition_sizes": {k: len(v) for k, v in parts.items()}, "excluded_count": len(result.excluded)}

    def add(code, severity, cols, rows, message, action, details=None):
        affected = sorted(set(int(i) for i in rows))
        evidence = dict(details or {})
        evidence.update(affected_count=len(affected), row_examples=affected[:20],
                        raw_examples=data.iloc[affected[:20]][cols].to_dict("records"),
                        examples_truncated=len(affected) > 20 or evidence.get("examples_truncated", False))
        findings.append(Finding(code, severity, cols, affected, message, action, evidence))

    active = data.index.difference(result.excluded)
    pair_metrics = {f"{a}:{b}": {"groups": {}, "exact_duplicate_pairs": 0} for a, b in combinations(parts, 2)}
    for a, b in combinations(parts, 2):
        key = f"{a}:{b}"
        for col in cfg.group_columns:
            left = data.iloc[parts[a]][col]
            right = data.iloc[parts[b]][col]
            shared = set(left.loc[~missing(left)]) & set(right.loc[~missing(right)])
            pair_metrics[key]["groups"][col] = len(shared)
            if shared:
                rows = [i for i in parts[a] + parts[b] if data.at[i, col] in shared]
                add("group_overlap", "error", [col], rows, f"Declared groups overlap between {a} and {b}.",
                    "Keep independent experiments or sources within one partition.", {"partitions": [a, b], "group_count": len(shared)})
            absent = [i for i in parts[a] + parts[b] if missing(data[col].iloc[[i]]).iloc[0]]
            if absent:
                add("group_overlap_not_evaluable", "warning", [col], absent,
                    "Missing identifiers prevent a complete group-overlap check.", "Recover group metadata.")
    for _, group in data.loc[active].groupby(columns, dropna=False, sort=False):
        by_part = {p: [int(i) for i in group.index if labels[i] == p] for p in parts}
        for a, b in combinations(parts, 2):
            if by_part[a] and by_part[b]:
                count = len(by_part[a]) * len(by_part[b])
                pair_metrics[f"{a}:{b}"]["exact_duplicate_pairs"] += count
                add("exact_duplicate_overlap", "error", columns, by_part[a] + by_part[b],
                    f"Exact records overlap between {a} and {b}.", "Resolve duplicate provenance and partition related records together.",
                    {"partitions": [a, b], "pair_count": count})
    metrics["pairwise_overlap"] = pair_metrics

    def summarize_pairs(pairs, method, cols):
        counts = {key: 0 for key in pair_metrics}
        affected, examples = set(), []
        for item in pairs:
            i, j = item[:2]
            a, b = labels[i], labels[j]
            if a == b or "excluded" in (a, b):
                continue
            key = f"{a}:{b}" if f"{a}:{b}" in counts else f"{b}:{a}"
            counts[key] += 1
            affected.update((i, j))
            if len(examples) < 100:
                examples.append(list(item))
        metrics[method] = counts
        if affected:
            add("near_duplicate_overlap", "warning", cols, affected,
                "Similar samples occur in different partitions under the configured criterion.",
                "Investigate chemical and experimental relationships or enable similarity grouping.",
                {"method": method, "counts": counts, "pair_examples": examples, "examples_truncated": sum(counts.values()) > len(examples)})

    if cfg.near_duplicates:
        cols = list(cfg.near_duplicates)
        values = matrix(data, cols)
        summarize_pairs(numeric_pairs(values, list(cfg.near_duplicates.values()), cfg.max_pair_comparisons), "numeric_near_duplicate_pairs", cols)
        invalid = [int(i) for i in active if not np.isfinite(values[i]).all()]
        if invalid:
            add("similarity_not_evaluable", "warning", cols, invalid, "Invalid numeric evidence skipped in similarity comparisons.", "Recover the measurements.")
    if cfg.smiles_column:
        _, fps, invalid = molecules(data[cfg.smiles_column])
        if cfg.molecular_similarity is not None:
            summarize_pairs(molecular_pairs(fps, cfg.molecular_similarity, cfg.max_pair_comparisons), "molecular_near_duplicate_pairs", [cfg.smiles_column])
        if invalid:
            add("molecular_similarity_not_evaluable", "warning", [cfg.smiles_column], set(invalid) & set(active),
                "Invalid or disconnected SMILES cannot support molecular comparisons.", "Validate molecular structures.")
    metrics["distribution_shift"] = _shifts(data, parts, cfg, add)
    metrics["training_domain"] = _domain(data, parts, cfg, add) if cfg.domain_columns else {"status": "not_configured"}
    metrics["checks_not_configured"] = [name for name, enabled in (
        ("numeric_similarity", cfg.near_duplicates), ("molecular_similarity", cfg.molecular_similarity),
        ("numeric_shift", cfg.numeric_columns), ("categorical_shift", cfg.categorical_columns), ("training_domain", cfg.domain_columns)) if not enabled]
    return metrics, findings


def _shifts(data, parts, cfg, add):
    reports = {}
    for name, rows in parts.items():
        if name == "train":
            continue
        report = {"numeric": {}, "categorical": {}}
        for col in cfg.numeric_columns:
            all_values = numeric(data[col])
            x, y = all_values[parts["train"]], all_values[rows]
            left, right = x[np.isfinite(x)], y[np.isfinite(y)]
            entry = {"train_invalid_count": int((~np.isfinite(x)).sum()), "evaluation_invalid_count": int((~np.isfinite(y)).sum())}
            if not len(left) or not len(right):
                entry["status"] = "not_evaluable"
            else:
                grid = np.sort(np.concatenate([left, right]))
                ks = float(np.max(np.abs(np.searchsorted(np.sort(left), grid, side="right") / len(left)
                                             - np.searchsorted(np.sort(right), grid, side="right") / len(right))))
                scale = float(np.std(left))
                entry.update(status="evaluated", ks_statistic=ks, train_mean=float(np.mean(left)), evaluation_mean=float(np.mean(right)),
                             standardized_mean_difference=float((np.mean(right) - np.mean(left)) / scale) if scale > 0 else None,
                             train_constant=scale == 0)
                if ks >= cfg.shift_threshold:
                    add("numeric_distribution_shift", "warning", [col], rows,
                        f"{name} differs from training in an empirical numeric distribution.",
                        "Interpret the model score within the observed shift; this threshold is descriptive, not a significance test.",
                        {"partition": name, **entry, "threshold": cfg.shift_threshold})
            invalid = [i for i in parts["train"] + rows if not np.isfinite(all_values[i])]
            if invalid:
                add("numeric_shift_not_evaluable", "warning", [col], invalid,
                    "Invalid measurements are excluded from numeric shift statistics.", "Inspect missingness and recover measurements.")
            report["numeric"][col] = entry
        for col in cfg.categorical_columns:
            def counts(positions):
                result = {}
                for i in positions:
                    v = data.at[i, col]
                    key = ("missing",) if missing(data[col].iloc[[i]]).iloc[0] else ("value", type(v).__name__, str(v))
                    result[key] = result.get(key, 0) + 1
                return result
            a, b = counts(parts["train"]), counts(rows)
            support = set(a) | set(b)
            tv = float(sum(abs(a.get(k, 0) / len(parts["train"]) - b.get(k, 0) / len(rows)) for k in support) / 2)
            unseen = set(b) - set(a) - {("missing",)}
            report["categorical"][col] = {"total_variation": tv, "unseen_categories": [list(k[1:]) for k in sorted(unseen)],
                                           "unseen_row_fraction": sum(b[k] for k in unseen) / len(rows)}
            if tv >= cfg.shift_threshold or unseen:
                add("categorical_distribution_shift", "warning", [col], rows,
                    f"{name} has changed category frequencies or unseen categories.", "Check transfer across the affected categories.",
                    {"partition": name, **report["categorical"][col], "threshold": cfg.shift_threshold})
        reports[name] = report
    return reports


def _domain(data, parts, cfg, add):
    values = matrix(data, cfg.domain_columns)
    valid = np.isfinite(values).all(axis=1)
    training = np.array([i for i in parts["train"] if valid[i]], dtype=int)
    evaluated = [i for p, rows in parts.items() if p != "train" for i in rows]
    unknown = [i for rows in parts.values() for i in rows if not valid[i]]
    if unknown:
        add("training_domain_not_evaluable", "warning", cfg.domain_columns, unknown,
            "Invalid rows cannot support numeric training-domain distances.", "Recover values; do not fit imputers on held-out data.")
    if not len(training):
        add("training_domain_not_evaluable", "warning", cfg.domain_columns, evaluated,
            "No complete training rows are available to define a domain.", "Supply valid training measurements.")
        return {"status": "not_evaluable", "reason": "No finite training rows."}
    x = values[training]
    scale = (np.array([cfg.domain_scales[c] for c in cfg.domain_columns]) if cfg.domain_scales else np.std(x, axis=0))
    constant = scale == 0
    scale = np.where(constant, 1, scale)
    z = x / scale
    lower, upper = x.min(axis=0), x.max(axis=0)
    cutoff = cfg.domain_distance_threshold
    if cutoff is None and len(x) >= 2:
        pair_budget(len(x), cfg.max_pair_comparisons)
        nearest = np.full(len(x), np.inf)
        for i in range(len(x) - 1):
            distances = np.linalg.norm(z[i + 1:] - z[i], axis=1)
            nearest[i] = min(nearest[i], float(distances.min()))
            nearest[i + 1:] = np.minimum(nearest[i + 1:], distances)
        cutoff = float(np.quantile(nearest, cfg.domain_quantile))
    pair_budget(len(training), cfg.max_pair_comparisons, len(evaluated))
    records, outside = [], []
    for i in evaluated:
        if not valid[i]:
            continue
        distances = np.linalg.norm(z - values[i] / scale, axis=1)
        j = int(np.argmin(distances))
        envelope = bool(((values[i] < lower) | (values[i] > upper)).any())
        distant = bool(distances[j] > cutoff) if cutoff is not None else None
        record = {"row": i, "nearest_training_row": int(training[j]), "distance": float(distances[j]),
                  "outside_training_range": envelope, "beyond_distance_threshold": distant}
        records.append(record)
        if envelope or distant:
            outside.append(i)
    if outside:
        add("outside_training_domain", "warning", cfg.domain_columns, outside,
            "Samples are outside training ranges or beyond a training-calibrated neighbor distance.",
            "Report performance for these samples separately; this geometric screen is not calibrated predictive uncertainty.",
            {"distance_threshold": cutoff, "calibration": "explicit" if cfg.domain_distance_threshold is not None else "training_leave_one_out_quantile"})
    if cutoff is None:
        add("training_domain_distance_not_evaluable", "info", cfg.domain_columns, evaluated,
            "At least two valid training rows are needed to calibrate neighbor distance.", "Supply an explicit distance threshold or more training data.")
    return {"status": "evaluated", "fit_rows": training.tolist(), "scale_source": "explicit" if cfg.domain_scales else "training_only_std",
            "scales": scale.tolist(), "constant_training_dimensions": [cfg.domain_columns[i] for i in np.flatnonzero(np.ptp(x, axis=0) == 0)],
            "lower": lower.tolist(), "upper": upper.tolist(), "distance_threshold": cutoff, "rows": records,
            "outside_rows": outside, "outside_fraction": len(outside) / len(records) if records else None}
