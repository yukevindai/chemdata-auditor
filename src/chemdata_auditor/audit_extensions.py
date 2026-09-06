"""Domain-aware checks and strict validation for the expanded audit configuration."""

import re

import numpy as np
import pandas as pd

from .common import finite_number, missing, names, numeric, require
from .rules import check_constraints, validate_predicate
from .similarity import matrix, molecular_pairs, molecules, numeric_pairs, pair_budget


def validate(cfg):
    for key in ("numeric_columns", "identity_columns", "conflict_columns", "identifier_columns", "density_columns"):
        names(getattr(cfg, key), key)
    if not isinstance(cfg.check_missing, bool):
        raise ValueError("check_missing must be boolean.")
    for key in ("conflict_tolerance", "density_radius"):
        finite_number(getattr(cfg, key), key)
        if getattr(cfg, key) < 0:
            raise ValueError(f"{key} must be nonnegative.")
    finite_number(cfg.correlation_threshold, "correlation_threshold")
    if not 0 < cfg.correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be in (0, 1].")
    for key in ("max_pair_comparisons", "min_neighbors"):
        v = getattr(cfg, key)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("near_duplicates", "density_scales", "provenance_patterns", "unit_aliases"):
        value = getattr(cfg, key)
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be an object.")
        names(list(value), key)
    for key in ("near_duplicates", "density_scales"):
        for value in getattr(cfg, key).values():
            finite_number(value, key)
            if value < 0 or (key == "density_scales" and value == 0):
                raise ValueError("Tolerances must be nonnegative and scales positive.")
    if set(cfg.density_columns) != set(cfg.density_scales):
        raise ValueError("density_scales must specify one positive scale per density column.")
    if cfg.conflict_columns and not cfg.identity_columns:
        raise ValueError("conflict_columns requires identity_columns defining one experimental observation.")
    for column, pattern in cfg.provenance_patterns.items():
        if not isinstance(pattern, str) or len(pattern) > 256:
            raise ValueError("Provenance patterns must be regex strings no longer than 256 characters.")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid provenance pattern for {column}.") from exc
    for value in cfg.unit_aliases.values():
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Unit aliases must map to explicit Pint unit strings.")
    if cfg.smiles_column is not None and (not isinstance(cfg.smiles_column, str) or not cfg.smiles_column):
        raise ValueError("smiles_column must be a column name.")
    if cfg.molecular_similarity is not None:
        finite_number(cfg.molecular_similarity, "molecular_similarity")
        if not cfg.smiles_column or not 0 < cfg.molecular_similarity <= 1:
            raise ValueError("molecular_similarity needs smiles_column and a threshold in (0, 1].")
    if not isinstance(cfg.compositions, list) or not isinstance(cfg.rules, list):
        raise ValueError("compositions and rules must be lists.")
    for comp in cfg.compositions:
        if not isinstance(comp, dict) or set(comp) - {"columns", "total", "tolerance"} or "columns" not in comp:
            raise ValueError("Composition needs columns and optional total/tolerance.")
        names(comp["columns"], "composition columns")
        if not comp["columns"]:
            raise ValueError("Composition columns cannot be empty.")
        for k, default in (("total", 100), ("tolerance", 1e-6)):
            finite_number(comp.get(k, default), k)
            if comp.get(k, default) < 0 or (k == "total" and comp.get(k, default) == 0):
                raise ValueError("Composition total must be positive and tolerance nonnegative.")
    for rule in cfg.rules:
        if not isinstance(rule, dict) or set(rule) - {"name", "when", "assert", "severity", "action"}:
            raise ValueError("Unknown constraint fields.")
        if not isinstance(rule.get("name"), str) or not rule["name"] or "assert" not in rule:
            raise ValueError("Constraint requires name and assert.")
        if rule.get("severity", "error") not in {"error", "warning", "info"}:
            raise ValueError("Invalid constraint severity.")
        if "action" in rule and not isinstance(rule["action"], str):
            raise ValueError("Constraint action must be text.")
        validate_predicate(rule["assert"])
        if "when" in rule:
            validate_predicate(rule["when"])


def normalized_units(data, cfg, add):
    """Convert only the working copy; preserve raw evidence and input values."""
    import pint
    units = pint.UnitRegistry()
    result = data.copy()
    for column, rule in cfg.units.items():
        unit_col, expected = rule["column"], rule["expected"].strip()
        expected = cfg.unit_aliases.get(expected, expected)
        try:
            target = units.Unit(expected)
        except (pint.PintError, ValueError, TypeError) as exc:
            raise ValueError(f"Unknown expected unit for {column}: {expected}") from exc
        raw = numeric(data[column])
        converted = np.full(len(data), np.nan)
        if unit_col not in data:
            add("missing_unit_column", "error", [column, unit_col], np.ones(len(data), bool),
                "The configured unit metadata column is absent.", "Supply explicit unit metadata.")
            result[column] = converted
            continue
        absent = missing(data[unit_col]).to_numpy()
        if absent.any():
            add("missing_unit", "error", [column, unit_col], absent, "Unit metadata is missing.", "Recover units from the source.")
        mismatched = ~absent & data[unit_col].astype(str).str.strip().ne(rule["expected"].strip()).to_numpy()
        if mismatched.any():
            add("unit_mismatch", "warning", [column, unit_col], mismatched,
                "Unit labels differ from the declared canonical label.", "Review the conversion evidence before combining measurements.",
                {"expected": expected})
        for label in data.loc[~absent, unit_col].astype(str).unique():
            mask = data[unit_col].astype(str).eq(label).to_numpy() & ~absent
            source = cfg.unit_aliases.get(label.strip(), label.strip())
            try:
                quantity = units.Quantity(raw[mask], units.Unit(source)).to(target)
                converted[mask] = quantity.magnitude
                if source != expected:
                    examples = {str(i): float(converted[i]) for i in np.flatnonzero(mask & np.isfinite(converted))[:20]}
                    add("unit_conversion", "info", [column, unit_col], mask,
                        "Compatible measurements converted internally for numeric checks; input data is unchanged.",
                        "Use the declared canonical units in downstream preprocessing.",
                        {"from": source, "to": expected, "converted_examples": examples})
            except pint.DimensionalityError:
                add("incompatible_units", "error", [column, unit_col], mask,
                    "Unit dimensions do not match the configured measurement.", "Verify the quantity and its units in the original source.")
            except (pint.PintError, ValueError, TypeError):
                add("unknown_unit", "error", [column, unit_col], mask,
                    "Cannot interpret the declared unit.", "Supply an unambiguous Pint unit or an explicit unit_aliases mapping.")
        result[column] = converted
    return result


def extended_checks(data, normalized, cfg, add, ran, skipped):
    require(data, cfg.numeric_columns + cfg.identity_columns + cfg.conflict_columns + cfg.identifier_columns
            + cfg.density_columns + list(cfg.near_duplicates) + list(cfg.provenance_patterns)
            + ([cfg.smiles_column] if cfg.smiles_column else []))
    if cfg.check_missing:
        ran.append("missing_values")
        for column in data:
            mask = missing(data[column]).to_numpy()
            if mask.any():
                add("missing_values", "warning", [column], mask, "Missing or blank values in the dataset.",
                    "Check missingness mechanisms; fit any imputation using training data only.", {"fraction": float(mask.mean())})
    else:
        skipped["missing_values"] = "check_missing is false."
    ran.append("schema_quality")
    for col in data:
        known = ~missing(data[col])
        if known.any() and data.loc[known, col].nunique() == 1:
            add("constant_column", "info", [col], known, "Column has one observed value.",
                "Keep useful metadata; avoid treating a constant feature as predictive evidence.")
    for col in dict.fromkeys([*cfg.identifier_columns, *cfg.feature_columns]):
        known = ~missing(data[col])
        count = int(known.sum())
        ratio = data.loc[known, col].nunique() / max(1, count)
        id_name = bool(re.search(r"(^id$|_id$|^id_|index|identifier|uuid|barcode)", col, re.I))
        if count >= 3 and (col in cfg.identifier_columns or id_name) and ratio >= 0.9:
            add("suspicious_identifier", "warning", [col], known,
                "A nearly unique identifier may encode sample order or identity rather than chemistry.",
                "Keep as metadata or grouping information; investigate before using it as a model feature.", {"unique_fraction": ratio})
    if cfg.identity_columns and cfg.conflict_columns:
        ran.append("conflicting_samples")
        known = ~data[cfg.identity_columns].apply(lambda s: missing(s)).any(axis=1)
        for _, group in data.loc[known].groupby(cfg.identity_columns, dropna=False, sort=False):
            if len(group) < 2:
                continue
            conflicting = []
            for col in cfg.conflict_columns:
                measured = normalized.loc[group.index, col]
                observed = measured.loc[~missing(measured)]
                if len(observed) < 2:
                    continue
                vals = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                different = (float(np.ptp(vals)) > cfg.conflict_tolerance if np.isfinite(vals).all()
                             else observed.nunique() > 1)
                if different:
                    conflicting.append(col)
            if conflicting:
                mask = data.index.isin(group.index)
                add("conflicting_samples", "error", cfg.identity_columns + conflicting, mask,
                    "Records with the same declared observation identity disagree.",
                    "Resolve source versions, measurement conditions, and replicate identity; do not average automatically.",
                    {"absolute_tolerance": cfg.conflict_tolerance})
    else:
        skipped["conflicting_samples"] = "Configure identity_columns and conflict_columns."
    if cfg.target_column and cfg.feature_columns:
        target = pd.to_numeric(data[cfg.target_column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        for col in cfg.feature_columns:
            if col == cfg.target_column:
                continue
            x = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            valid = np.isfinite(x) & np.isfinite(target)
            if valid.sum() >= 5 and np.ptp(x[valid]) > 0 and np.ptp(target[valid]) > 0:
                corr = float(np.corrcoef(x[valid], target[valid])[0, 1])
                if np.isfinite(corr) and abs(corr) >= cfg.correlation_threshold:
                    add("target_proxy", "warning", [col, cfg.target_column], valid,
                        "Feature is strongly linearly associated with the target; this is a screening signal, not proof of leakage.",
                        "Check causal ordering, target-derived transforms, and availability at prediction time.",
                        {"pearson_r": corr, "threshold": cfg.correlation_threshold})
    if cfg.near_duplicates:
        ran.append("near_duplicates")
        columns = list(cfg.near_duplicates)
        values = matrix(normalized, columns)
        _pair_findings(data, cfg, add, numeric_pairs(values, list(cfg.near_duplicates.values()), cfg.max_pair_comparisons), columns, "numeric")
        invalid = ~np.isfinite(values).all(axis=1)
        if invalid.any():
            add("similarity_not_evaluable", "warning", columns, invalid, "Invalid measurements excluded from near-duplicate comparisons.", "Recover the measurements.")
    else:
        skipped["near_duplicates"] = "No per-column absolute tolerances configured."
    if cfg.smiles_column:
        ran.append("molecular_identity")
        identities, fingerprints, invalid = molecules(data[cfg.smiles_column])
        if invalid:
            add("invalid_molecule", "error", [cfg.smiles_column], data.index.isin(invalid),
                "Missing, invalid, or disconnected molecular structure.", "Validate SMILES and explicitly standardize salts/mixtures.")
        duplicates = pd.Series(identities).duplicated(keep=False) & pd.Series(identities).notna()
        if duplicates.any():
            add("duplicate_molecules", "warning", [cfg.smiles_column], duplicates,
                "Multiple records describe the same canonical isomeric molecule.", "Check experimental context and group repeated molecules when appropriate.")
        if cfg.molecular_similarity is not None:
            ran.append("molecular_similarity")
            _pair_findings(data, cfg, add, molecular_pairs(fingerprints, cfg.molecular_similarity, cfg.max_pair_comparisons),
                           [cfg.smiles_column], "morgan_tanimoto")
    else:
        skipped["molecular_identity"] = "No smiles_column configured."
    if cfg.density_columns:
        ran.append("multivariate_density")
        values = matrix(normalized, cfg.density_columns) / np.array([cfg.density_scales[c] for c in cfg.density_columns])
        pair_budget(len(values), cfg.max_pair_comparisons)
        finite = np.isfinite(values).all(axis=1)
        counts = np.zeros(len(values), dtype=int)
        for i in np.flatnonzero(finite):
            distances = np.linalg.norm(values[i + 1:] - values[i], axis=1)
            neighbors = np.flatnonzero(finite[i + 1:] & (distances <= cfg.density_radius)) + i + 1
            counts[i] += len(neighbors)
            counts[neighbors] += 1
        sparse = finite & (counts < cfg.min_neighbors)
        if sparse.any():
            add("sparse_neighborhood", "info", cfg.density_columns, sparse,
                "Few neighbors in the declared scaled numeric space.", "Inspect joint coverage and independence of repeated measurements.",
                {"radius": cfg.density_radius, "min_neighbors": cfg.min_neighbors,
                 "counts": {str(i): int(counts[i]) for i in np.flatnonzero(sparse)}})
        if (~finite).any():
            add("density_not_evaluable", "warning", cfg.density_columns, ~finite,
                "Invalid rows excluded from neighborhood counts.", "Recover measurements before assessing joint coverage.")
    else:
        skipped["multivariate_density"] = "No density_columns and density_scales configured."
    if cfg.compositions or cfg.rules:
        ran.append("domain_constraints")
        check_constraints(normalized, cfg, add)
    else:
        skipped["domain_constraints"] = "No compositions or rules configured."
    for col in set(cfg.provenance_columns) | set(cfg.provenance_patterns):
        if col not in data:
            continue
        vals = data[col].astype(str).str.strip()
        weak = vals.str.lower().isin({"unknown", "n/a", "na", "none", "todo", "tbd", "unspecified"}) & ~missing(data[col])
        if col in cfg.provenance_patterns:
            weak |= ~vals.str.fullmatch(cfg.provenance_patterns[col]) & ~missing(data[col])
        if weak.any():
            add("weak_provenance", "warning", [col], weak, "Provenance is a placeholder or fails the declared syntax rule.",
                "Recover a traceable source; syntax checks do not verify source authenticity.")


def _pair_findings(data, cfg, add, pairs, columns, method):
    count, cross_count, examples, cross_examples, rows, cross_rows = 0, 0, [], [], set(), set()
    for pair in pairs:
        i, j = pair[:2]
        count += 1
        rows.update((i, j))
        if len(examples) < 100:
            examples.append(list(pair))
        if cfg.split_column and not missing(data[cfg.split_column].iloc[[i, j]]).any() and data[cfg.split_column].iloc[i] != data[cfg.split_column].iloc[j]:
            cross_count += 1
            cross_rows.update((i, j))
            if len(cross_examples) < 100:
                cross_examples.append(list(pair))
    for code, number, affected, evidence, severity in (
        ("near_duplicate_samples", count, rows, examples, "warning"),
        ("near_duplicate_split_overlap", cross_count, cross_rows, cross_examples, "error"),
    ):
        if number:
            add(code, severity, columns, data.index.isin(affected), "Samples meet the declared similarity criterion.",
                "Review identities and measurement context; group related records before evaluating models.",
                {"method": method, "pair_count": number, "pair_examples": evidence, "examples_truncated": number > len(evidence)})
