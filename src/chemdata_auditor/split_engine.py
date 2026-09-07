"""Allocation engines, indivisible sample relationships, and scientific claim scope."""

from decimal import Decimal, InvalidOperation
from dataclasses import asdict
from itertools import combinations
import math
from numbers import Number
import random
import re

import numpy as np
import pandas as pd

from .audit import Finding
from .common import frame, metadata, require
from .diagnostics import DiagnosticsConfig, diagnose, validate_partition
from .similarity import matrix, molecules, molecular_pairs, numeric_pairs

CLAIMS = {
    "random": ("Performance on randomly allocated observations or declared independent groups from the sampled population.",
               "Does not establish transfer to new chemistry, laboratories, publications, or future experiments."),
    "formulation": ("Transfer to held-out formulation identities or declared formulation families.",
                    "Different labels may still describe chemically similar formulations; verify equivalence and provenance."),
    "composition": ("Transfer to compositions absent from training under declared representation and rounding.",
                    "Composition holdout alone does not establish extrapolation, novel scaffold transfer, or new-laboratory performance."),
    "scaffold": ("Transfer to held-out Bemis–Murcko molecular scaffold groups.",
                 "Scaffold disjointness does not ensure fingerprint dissimilarity, causal validity, or transfer to all chemical classes."),
    "publication": ("Transfer across held-out publication/source groups.",
                    "Publications can reuse datasets; source labels alone do not establish experimental independence."),
    "laboratory": ("Transfer to held-out laboratory environments.",
                   "Laboratory holdout may confound chemistry, instrumentation, and procedure; it does not isolate their causal effects."),
    "time": ("Performance on observations available after the declared training and validation cutoffs.",
             "Temporal order does not guarantee independence of repeated samples or resistance to later distribution drift."),
    "cluster": ("Transfer to held-out clusters in a declared covariate representation.",
                "Clusters use all supplied covariates to design the benchmark; this is not a prospective or causal guarantee."),
    "extrapolation": ("Performance beyond the declared numeric training/development region in selected dimensions.",
                      "This does not imply arbitrary chemical extrapolation or calibrated predictive uncertainty."),
}


def _composition(data, cfg):
    keys = []
    for row in data[cfg.columns].itertuples(index=False, name=None):
        key = []
        for value in row:
            try:
                number = Decimal(str(value))
            except InvalidOperation:
                key.append(("label", str(value)))
            else:
                if not number.is_finite():
                    raise ValueError("Composition values cannot be nonfinite.")
                if cfg.composition_decimals is not None:
                    try:
                        number = number.quantize(Decimal(1).scaleb(-cfg.composition_decimals))
                    except InvalidOperation as exc:
                        raise ValueError("Composition magnitude exceeds rounding precision; normalize units first.") from exc
                key.append(("number", number))
        keys.append(tuple(key))
    return keys


def _allocate(components, cfg):
    count = len(components)
    partitions = 3 if cfg.validation_size else 2
    if count < partitions:
        raise ValueError(f"At least {partitions} independent groups are required for nonempty partitions.")
    shuffled = list(components)
    random.Random(cfg.seed).shuffle(shuffled)
    n_test = min(count - partitions + 1, max(1, math.ceil(count * cfg.test_size)))
    n_val = min(count - n_test - 1, max(1, math.ceil(count * cfg.validation_size))) if cfg.validation_size else 0
    lookup = {}
    for i, group in enumerate(shuffled):
        label = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
        lookup.update({row: label for row in group})
    return [lookup[i] for i in range(sum(map(len, components)))], n_test, n_val


def _timestamps(series):
    if any(isinstance(value, Number) for value in series):
        raise ValueError("Numeric timestamps are ambiguous; supply ISO timestamps.")
    if any(isinstance(value, str) and not re.match(r"^\d{4}-\d{2}-\d{2}(?:T| |$)", value) for value in series):
        raise ValueError("Time values must use absolute ISO dates or timestamps.")
    values = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    if values.isna().any():
        raise ValueError("Invalid time values or cutoff.")
    return values


def _boundary(data, cfg, strategy):
    if strategy == "time":
        values = _timestamps(data[cfg.columns[0]])
        cutoff = _timestamps(pd.Series([cfg.cutoff])).iloc[0]
        validation = _timestamps(pd.Series([cfg.validation_cutoff])).iloc[0] if cfg.validation_cutoff else None
        if validation is not None and validation >= cutoff:
            raise ValueError("validation_cutoff must precede cutoff.")
        labels = ["test" if v > cutoff else "validation" if validation is not None and v > validation else "train" for v in values]
        return labels, {"cutoff_utc": cutoff.isoformat(), "validation_cutoff_utc": validation.isoformat() if validation is not None else None,
                        "boundary_equality": "earlier partition"}, values.to_frame()
    values = matrix(data, cfg.columns)
    if not np.isfinite(values).all():
        raise ValueError("Extrapolation values must be finite numeric measurements.")
    if cfg.extrapolation_bounds:
        lower = np.array([cfg.extrapolation_bounds[c][0] for c in cfg.columns])
        upper = np.array([cfg.extrapolation_bounds[c][1] for c in cfg.columns])
        test = ((values < lower) | (values > upper)).any(axis=1)
        val = np.zeros(len(data), bool)
        if cfg.validation_bounds:
            inner_lo = np.array([cfg.validation_bounds[c][0] for c in cfg.columns])
            inner_hi = np.array([cfg.validation_bounds[c][1] for c in cfg.columns])
            val = ~test & ((values < inner_lo) | (values > inner_hi)).any(axis=1)
        boundary = {"development_bounds": cfg.extrapolation_bounds, "training_bounds": cfg.validation_bounds or cfg.extrapolation_bounds,
                    "equality": "inside"}
    else:
        x = values[:, 0]
        test = x > cfg.threshold if cfg.direction == "high" else x < cfg.threshold
        val = np.zeros(len(data), bool)
        if cfg.validation_threshold is not None:
            val = ~test & ((x > cfg.validation_threshold) if cfg.direction == "high" else (x < cfg.validation_threshold))
        boundary = {"threshold": cfg.threshold, "validation_threshold": cfg.validation_threshold,
                    "direction": cfg.direction, "equality": "inner partition"}
    return ["test" if test[i] else "validation" if val[i] else "train" for i in range(len(data))], boundary, pd.DataFrame(values, columns=cfg.columns)


def run_split(data, cfg):
    from .split import SplitResult, _components, _scaffolds
    cfg.__post_init__()
    data = frame(data)
    require(data, cfg.columns + cfg.group_columns + list(cfg.group_near_duplicates)
            + ([cfg.group_smiles_column] if cfg.group_smiles_column else [])
            + ([cfg.target_column] if cfg.target_column else []), complete=True)
    strategy = "time" if cfg.strategy == "temporal" else cfg.strategy
    n, notes, cut_values, keys = len(data), [], None, None
    key_sets = [data[c].tolist() for c in cfg.group_columns]
    edges = []
    if cfg.group_near_duplicates:
        values = matrix(data, list(cfg.group_near_duplicates))
        if not np.isfinite(values).all():
            raise ValueError("Numeric similarity grouping requires finite measurements.")
        edges.extend(numeric_pairs(values, list(cfg.group_near_duplicates.values()), cfg.max_pair_comparisons))
        notes.append("Numeric near-duplicate relationships are transitive grouping constraints; long chains may join distant endpoints.")
    if cfg.group_smiles_column:
        canonical, fingerprints, invalid = molecules(data[cfg.group_smiles_column])
        if invalid:
            raise ValueError(f"Invalid or disconnected SMILES in molecular grouping at rows {invalid[:20]}.")
        key_sets.append(canonical)
        if cfg.group_molecular_similarity is not None:
            edges.extend((i, j) for i, j, _ in molecular_pairs(fingerprints, cfg.group_molecular_similarity, cfg.max_pair_comparisons))
    if not cfg.group_columns and not cfg.group_near_duplicates and not cfg.group_smiles_column:
        notes.append("No sample relationship constraints supplied; undeclared repeated experiments can still overlap.")
    info = {"strategy": strategy, "warnings": notes}
    if strategy in {"time", "extrapolation"}:
        labels, boundary, cut_values = _boundary(data, cfg, strategy)
        info["boundary"] = boundary
        components = _components(n, key_sets, edges)
        crossing = [g for g in components if len({labels[i] for i in g}) > 1]
        if crossing and cfg.crossing_policy == "error":
            raise ValueError(f"Sample groups cross the split boundary (rows {[i for g in crossing for i in g][:20]}). "
                             "Use crossing_policy='exclude' to exclude entire crossing groups.")
        for group in crossing:
            for i in group:
                labels[i] = "excluded"
        info["excluded_crossing_groups"] = len(crossing)
        if crossing:
            notes.append("Entire crossing groups were excluded; assess selection bias and reduced coverage.")
    else:
        if strategy == "composition":
            keys = _composition(data, cfg)
            notes.append("Numeric composition representations are normalized by decimal value; rounding is explicit. Units and chemical equivalence are not inferred.")
        elif strategy == "scaffold":
            keys = _scaffolds(data[cfg.columns[0]])
            notes.append("Scaffolds ignore chirality and group all acyclic molecules together; no tautomer normalization is performed.")
        elif strategy in {"formulation", "publication", "laboratory"}:
            keys = data[cfg.columns[0]].tolist()
        elif strategy == "cluster":
            from sklearn.cluster import KMeans
            values = matrix(data, cfg.columns)
            if not np.isfinite(values).all():
                raise ValueError("Clustering requires finite numeric covariates.")
            values = values / np.array([cfg.cluster_scales[c] for c in cfg.columns])
            if len(np.unique(values, axis=0)) < cfg.n_clusters:
                raise ValueError("n_clusters exceeds the number of distinct covariate vectors.")
            model = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.seed, n_init=10, max_iter=300)
            keys = model.fit_predict(values).tolist()
            info["cluster_design"] = {"centers_scaled": model.cluster_centers_.tolist(), "inertia": float(model.inertia_),
                                       "fit_scope": "all supplied covariates for benchmark design only", "labels": keys}
            notes.append("K-means uses all covariates for partition design with explicit scales. Do not reuse this fit as training-only model preprocessing.")
        if keys is not None:
            key_sets.insert(0, keys)
        components = _components(n, key_sets, edges)
        if cfg.holdout_values is not None:
            requested = set(cfg.holdout_values) | set(cfg.validation_values or [])
            unknown = requested - set(keys)
            if unknown:
                prefix = "Unknown laboratory" if strategy == "laboratory" else "Unknown"
                raise ValueError(f"{prefix} holdout values: {sorted(map(str, unknown))}")
            labels = ["test" if key in cfg.holdout_values else "validation" if key in (cfg.validation_values or []) else "train" for key in keys]
            if any(len({labels[i] for i in g}) > 1 for g in components):
                raise ValueError("Declared sample groups connect held-out identities to other partitions.")
            info["selection"] = "explicit scientific identities"
        else:
            labels, test_groups, val_groups = _allocate(components, cfg)
            info.update(selection="seeded connected-group holdout", requested_test_group_fraction=cfg.test_size,
                        requested_validation_group_fraction=cfg.validation_size, heldout_components=test_groups, validation_components=val_groups)
        if keys is not None:
            info["scientific_groups"] = len(set(keys))
    info["independent_components"] = len(components)
    partitions = {p: [i for i, label in enumerate(labels) if label == p] for p in ("train", "validation", "test", "excluded")}
    want_validation = bool(cfg.validation_size or cfg.validation_values or cfg.validation_cutoff or cfg.validation_threshold is not None or cfg.validation_bounds)
    if not partitions["train"] or not partitions["test"] or (want_validation and not partitions["validation"]):
        raise ValueError("The requested split leaves an empty training, validation, or test partition.")
    active = {p: rows for p, rows in partitions.items() if rows and p != "excluded"}
    overlap = {c: sum(len(set(data.iloc[active[a]][c]) & set(data.iloc[active[b]][c])) for a, b in combinations(active, 2)) for c in cfg.group_columns}
    info["group_overlap_counts"] = overlap
    if keys is not None:
        info["scientific_group_overlap"] = sum(len({keys[i] for i in active[a]} & {keys[i] for i in active[b]}) for a, b in combinations(active, 2))
    if any(overlap.values()) or info.get("scientific_group_overlap", 0):
        raise RuntimeError("Internal invariant failed: scientific groups overlap partitions.")
    if cut_values is not None:
        ranges = {}
        for p, rows in active.items():
            ranges[p] = {c: ([cut_values.iloc[rows][c].min().isoformat(), cut_values.iloc[rows][c].max().isoformat()] if strategy == "time"
                             else [float(cut_values.iloc[rows][c].min()), float(cut_values.iloc[rows][c].max())]) for c in cut_values.columns}
        info["observed_ranges"] = ranges
    for p, rows in partitions.items():
        info[f"n_{p}"] = len(rows)
    count = sum(map(len, active.values()))
    info["actual_test_row_fraction"] = len(partitions["test"]) / count
    info["actual_validation_row_fraction"] = len(partitions["validation"]) / count
    info["generalization"] = {"supports": CLAIMS[strategy][0], "does_not_establish": CLAIMS[strategy][1],
                              "conditional_on": "Sample independence, provenance, measurement validity, and absence of target leakage must be checked separately."}
    if cfg.allow_target_in_split and cfg.target_column:
        notes.append("Target-dependent benchmark design is explicitly allowed; disclose outcome-conditioned selection and avoid using this as an unbiased deployment estimate.")
    result = SplitResult(partitions["train"], partitions["test"], partitions["excluded"], info, metadata(data, cfg), validation=partitions["validation"])
    validate_partition(result, n)
    diag_config = dict(cfg.diagnostics)
    diag_config["group_columns"] = list(dict.fromkeys([*cfg.group_columns, *diag_config.get("group_columns", [])]))
    diagnostics, findings = diagnose(data, result, DiagnosticsConfig(**diag_config))
    result.diagnostics["evaluation"] = diagnostics
    result.metadata["effective_diagnostics_config"] = asdict(DiagnosticsConfig(**diag_config))
    result.findings = findings
    result.findings.append(Finding("generalization_scope", "info", cfg.columns, partitions["validation"] + partitions["test"],
                                  CLAIMS[strategy][0], "Report this evaluation scope and its limitations alongside model metrics.", info["generalization"]))
    if partitions["excluded"]:
        result.findings.append(Finding("boundary_group_exclusion", "warning", cfg.group_columns, partitions["excluded"],
                                      "Entire related groups were excluded to preserve scientific boundaries.",
                                      "Investigate selection bias before interpreting performance.", {"excluded_groups": info["excluded_crossing_groups"]}))
    return result
