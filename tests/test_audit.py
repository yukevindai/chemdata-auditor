import json

import numpy as np
import pandas as pd
import pytest

from chemdata_auditor import AuditConfig, audit


def codes(report):
    return {f.code for f in report.findings}


def test_all_six_categories_and_positional_rows_without_mutation():
    df = pd.DataFrame({
        "sample": ["a", "a", "b", "c"], "partition": ["train", "test", "train", "test"],
        "temperature": [300, 300, -1, 500], "unit": ["K", "K", "C", ""],
        "source": ["paper1", "paper1", "", None],
    }, index=[8, 8, 20, 40])
    original = df.copy(deep=True)
    report = audit(df, AuditConfig(duplicate_columns=["sample", "temperature"],
        group_columns=["sample"], split_column="partition", bounds={"temperature": [0, 600]},
        units={"temperature": {"column": "unit", "expected": "K"}},
        provenance_columns=["source"], sparse_bins={"temperature": [0, 250, 400, 600]}, min_bin_count=2))
    assert {"duplicate_samples", "duplicate_split_overlap", "group_split_overlap", "out_of_bounds",
            "unit_mismatch", "missing_unit", "provenance_gap", "sparse_bin"} <= codes(report)
    overlap = next(f for f in report.findings if f.code == "group_split_overlap")
    assert overlap.rows == [0, 1]
    assert any(f.code == "sparse_bin" and f.rows == [] for f in report.findings)
    pd.testing.assert_frame_equal(df, original)
    json.loads(report.to_json())


def test_default_duplicate_comparison_excludes_partition():
    report = audit(pd.DataFrame({"x": [1, 1], "split": ["train", "validation"]}), AuditConfig(split_column="split"))
    assert "duplicate_split_overlap" in codes(report)
    assert "units" in report.checks_skipped


def test_invalid_measurements_and_inclusive_bounds():
    report = audit(pd.DataFrame({"x": [0, 1, np.inf, "bad", None, -1]}), AuditConfig(bounds={"x": [0, 1]}))
    assert next(f for f in report.findings if f.code == "invalid_numeric").rows == [2, 3, 4]
    assert next(f for f in report.findings if f.code == "out_of_bounds").rows == [5]


def test_missing_provenance_and_unit_columns_are_findings():
    report = audit(pd.DataFrame({"x": [1]}), AuditConfig(provenance_columns=["doi"], units={"x": {"column": "unit", "expected": "K"}}))
    assert codes(report) == {"provenance_gap", "missing_unit_column"}


def test_sparse_bins_final_endpoint_counted_once_and_outside_reported():
    report = audit(pd.DataFrame({"x": [-1, 0, 1, 2, 3]}), AuditConfig(sparse_bins={"x": [0, 1, 2]}, min_bin_count=3))
    bins = [f for f in report.findings if f.code == "sparse_bin"]
    assert [f.details["count"] for f in bins] == [1, 2]
    assert next(f for f in report.findings if f.code == "outside_sparse_bins").rows == [0, 4]


def test_feature_leakage_is_explicit_and_target_copy_is_only_a_warning():
    df = pd.DataFrame({"target": [1, 2, 3], "copy": [1, 2, 3], "future": [4, 5, 6]})
    report = audit(df, AuditConfig(feature_columns=["target", "copy", "future"],
                                 target_column="target", unavailable_features=["future"]))
    assert sum(f.code == "unavailable_feature" for f in report.findings) == 2
    assert next(f for f in report.findings if f.code == "target_copy").severity == "warning"


def test_missing_groups_do_not_create_fictitious_group_overlap():
    df = pd.DataFrame({"group": [None, "", "a", "a"], "x": [1, 2, 3, 4], "split": ["train", "test", "train", ""]})
    report = audit(df, AuditConfig(group_columns=["group"], split_column="split"))
    assert {"missing_group", "missing_split"} <= codes(report)
    assert "group_split_overlap" not in codes(report)


@pytest.mark.parametrize("kwargs", [
    {"bounds": {"x": [2, 1]}}, {"bounds": {"x": [None, None]}}, {"bounds": {"x": [0, float('inf')]}},
    {"sparse_bins": {"x": [0, 0]}}, {"units": {"x": {"expected": "K"}}},
    {"min_bin_count": True}, {"duplicate_columns": []}, {"group_columns": "sample"},
    {"split_column": "split", "duplicate_columns": ["x", "split"]},
    {"split_column": "split", "group_columns": ["split"]},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        AuditConfig(**kwargs)


def test_missing_configured_measurement_column_is_error():
    with pytest.raises(ValueError, match="Missing required"):
        audit(pd.DataFrame({"x": [1]}), AuditConfig(bounds={"typo": [0, 1]}))
