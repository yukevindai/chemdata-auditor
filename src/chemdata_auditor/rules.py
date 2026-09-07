"""Declarative constraints. Configuration is data: no eval, expressions, or imports."""

import operator
import re

import numpy as np
import pandas as pd

from .common import finite_number, missing, require
from .similarity import matrix

OPS = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le,
       "gt": operator.gt, "ge": operator.ge}


def validate_predicate(rule, depth=0):
    if depth > 8 or not isinstance(rule, dict):
        raise ValueError("Predicates must be objects nested at most eight levels.")
    for logical in ("all", "any"):
        if logical in rule:
            if set(rule) != {logical} or not isinstance(rule[logical], list) or not rule[logical]:
                raise ValueError("all/any needs a nonempty list of predicates.")
            for child in rule[logical]:
                validate_predicate(child, depth + 1)
            return
    if not isinstance(rule.get("column"), str) or not rule["column"]:
        raise ValueError("A predicate requires a column.")
    if rule.get("op") not in {*OPS, "in", "not_in", "before", "after", "on_or_before", "on_or_after"}:
        raise ValueError("Unsupported predicate operator.")
    if set(rule) - {"column", "op", "value", "other_column", "kind"}:
        raise ValueError("Unknown predicate fields.")
    if ("value" in rule) == ("other_column" in rule):
        raise ValueError("Supply exactly one value or other_column.")
    if "other_column" in rule and (not isinstance(rule["other_column"], str) or not rule["other_column"]):
        raise ValueError("other_column must be a name.")
    if rule.get("kind", "numeric") not in {"numeric", "string"}:
        raise ValueError("Predicate kind must be numeric or string.")
    if rule["op"] in {"in", "not_in"}:
        if "value" not in rule or not isinstance(rule["value"], list) or not rule["value"]:
            raise ValueError("in/not_in requires a nonempty value list.")
    if "value" in rule:
        vals = rule["value"] if isinstance(rule["value"], list) else [rule["value"]]
        if any(not isinstance(v, (str, int, float, bool)) for v in vals):
            raise ValueError("Predicate values must be scalar strings or numbers.")
        if rule.get("kind", "numeric") == "numeric" and rule["op"] in {*OPS, "in", "not_in"}:
            for value in vals:
                if isinstance(value, bool):
                    raise ValueError("Numeric predicate constants cannot be booleans.")
                try:
                    converted = float(value)
                except (ValueError, TypeError, OverflowError) as exc:
                    raise ValueError("Numeric predicate constants must be finite numbers.") from exc
                finite_number(converted, "Numeric predicate constant")
        if rule["op"] not in {"in", "not_in"} and isinstance(rule["value"], list):
            raise ValueError("Only membership predicates accept list values.")


def evaluate(data, rule):
    """Return (true, known) masks with three-valued logic for missing evidence."""
    if "all" in rule or "any" in rule:
        mode = "all" if "all" in rule else "any"
        results = [evaluate(data, r) for r in rule[mode]]
        true = np.vstack([t & k for t, k in results])
        false = np.vstack([~t & k for t, k in results])
        if mode == "all":
            return true.all(axis=0), true.all(axis=0) | false.any(axis=0)
        return true.any(axis=0), true.any(axis=0) | false.all(axis=0)
    col, op = rule["column"], rule["op"]
    require(data, [col] + ([rule["other_column"]] if "other_column" in rule else []))
    left = data[col]
    right = data[rule["other_column"]] if "other_column" in rule else pd.Series([rule["value"]] * len(data))
    if op in {"in", "not_in"}:
        known = ~missing(left).to_numpy()
        choices = rule["value"]
        if rule.get("kind", "numeric") == "numeric":
            left = pd.to_numeric(left, errors="coerce")
            choices = pd.to_numeric(pd.Series(choices), errors="coerce")
            if choices.isna().any():
                raise ValueError("Numeric membership needs numeric choices; use kind='string' for categories.")
            known &= np.isfinite(left.to_numpy(dtype=float, na_value=np.nan))
        else:
            left, choices = left.astype(str), [str(x) for x in choices]
        result = left.isin(choices).to_numpy()
        return result if op == "in" else ~result, known
    known = (~missing(left) & ~missing(right)).to_numpy()
    if op in {"before", "after", "on_or_before", "on_or_after"}:
        absolute = lambda value: isinstance(value, pd.Timestamp) or (isinstance(value, str) and bool(re.match(r"^\d{4}-\d{2}-\d{2}(?:T| |$)", value)))
        known &= (left.map(absolute) & right.map(absolute)).to_numpy()
        left, right = left.where(known), right.where(known)
        left = pd.to_datetime(left, utc=True, format="mixed", errors="coerce")
        right = pd.to_datetime(right, utc=True, format="mixed", errors="coerce")
        known &= (left.notna() & right.notna()).to_numpy()
        op = {"before": "lt", "after": "gt", "on_or_before": "le", "on_or_after": "ge"}[op]
    elif rule.get("kind", "numeric") == "numeric":
        left, right = pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")
        known &= np.isfinite(left.to_numpy(dtype=float, na_value=np.nan)) & np.isfinite(right.to_numpy(dtype=float, na_value=np.nan))
    else:
        left, right = left.astype(str), right.astype(str)
    return np.asarray(OPS[op](left, right).fillna(False), bool), known


def check_constraints(data, cfg, add):
    for rule in cfg.rules:
        condition, known_condition = (evaluate(data, rule["when"]) if "when" in rule
                                      else (np.ones(len(data), bool), np.ones(len(data), bool)))
        passed, known = evaluate(data, rule["assert"])
        bad = known_condition & condition & known & ~passed
        unknown = ~known_condition | (condition & ~known)
        columns = sorted(_columns(rule["assert"]) | (_columns(rule["when"]) if "when" in rule else set()))
        if bad.any():
            add("constraint_violation", rule.get("severity", "error"), columns, bad, rule["name"],
                rule.get("action", "Review the source measurement and the applicability of this rule."), {"rule": rule})
        if unknown.any():
            add("constraint_not_evaluable", "warning", columns, unknown,
                f"Insufficient evidence to evaluate: {rule['name']}", "Recover missing or invalid inputs.", {"rule": rule})
    for comp in cfg.compositions:
        columns, total, tolerance = comp["columns"], comp.get("total", 100), comp.get("tolerance", 1e-6)
        values = matrix(data, columns)
        valid = np.isfinite(values).all(axis=1)
        negative = (values < 0).any(axis=1)
        sums = values.sum(axis=1)
        bad = valid & (negative | (np.abs(sums - total) > tolerance))
        if bad.any():
            add("composition_error", "error", columns, bad, "Composition violates nonnegativity or the declared total.",
                "Check the composition basis, units, omitted components, and rounding.",
                {"total": total, "absolute_tolerance": tolerance, "sums": {str(i): float(sums[i]) for i in np.flatnonzero(bad)}})
        if (~valid).any():
            add("composition_not_evaluable", "warning", columns, ~valid, "Composition contains invalid or missing values.",
                "Recover the missing components before interpreting the sum.")


def _columns(rule):
    if "all" in rule or "any" in rule:
        return set().union(*[_columns(c) for c in rule.get("all", rule.get("any"))])
    return {rule["column"]} | ({rule["other_column"]} if "other_column" in rule else set())
