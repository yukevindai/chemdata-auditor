import pandas as pd
import pytest

from chemdata_auditor import AuditConfig, audit


@pytest.mark.parametrize("required", [[], ["doi"]])
def test_missing_patterned_provenance_preserves_report(required):
    report = audit(pd.DataFrame({"x": [1, 2]}), AuditConfig(provenance_columns=required, provenance_patterns={"doi": r"10\.\d+/.+"}))
    gap = next(f for f in report.findings if f.code == "provenance_gap")
    assert gap.columns == ["doi"] and gap.rows == [0, 1]
    assert "provenance" in report.checks_run


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "NaN", "Infinity", 10**400])
@pytest.mark.parametrize("op", ["ge", "in"])
def test_nonfinite_predicate_constants_rejected(value, op):
    with pytest.raises(ValueError, match="finite"):
        AuditConfig(rules=[{"name": "finite rule", "assert": {"column": "x", "op": op, "value": [value] if op == "in" else value}}])


def test_finite_numeric_string_constants_remain_supported():
    report = audit(pd.DataFrame({"x": [0, 2]}), AuditConfig(rules=[
        {"name": "positive", "assert": {"column": "x", "op": "gt", "value": "1"}}]))
    assert next(f for f in report.findings if f.code == "constraint_violation").rows == [0]
