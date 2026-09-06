import json

import pandas as pd
import pytest

from chemdata_auditor import SplitConfig, split


def assert_partition(result, n):
    groups = [set(result.train), set(result.test), set(result.excluded)]
    assert set.union(*groups) == set(range(n))
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert result.train and result.test
    assert len(result.assignments()) == n
    json.loads(result.to_json())


def test_composition_reproducible_and_groups_disjoint():
    df = pd.DataFrame({"ec": [10, 10, 20, 20, 30, 30], "salt": [1] * 6}, index=[5] * 6)
    cfg = SplitConfig("composition", ["ec", "salt"], test_size=0.34, seed=42)
    first, second = split(df, cfg), split(df, cfg)
    assert first == second
    assert_partition(first, len(df))
    assert not set(df.iloc[first.train].ec) & set(df.iloc[first.test].ec)
    assert first.diagnostics["scientific_group_overlap"] == 0


def test_transitive_groups_cannot_leak():
    df = pd.DataFrame({"formula": ["a", "b", "c", "d", "e"],
                       "sample": ["x", "x", "y", "z", "q"], "batch": ["p", "r", "r", "s", "t"]})
    result = split(df, SplitConfig("composition", ["formula"], group_columns=["sample", "batch"], test_size=0.5))
    labels = result.assignments()
    assert labels[0] == labels[1] == labels[2]
    assert all(count == 0 for count in result.diagnostics["group_overlap_counts"].values())
    assert_partition(result, len(df))


def test_laboratory_explicit_holdout_and_unknown_label():
    df = pd.DataFrame({"lab": ["A", "B", "A", "C"]})
    result = split(df, SplitConfig("laboratory", ["lab"], holdout_values=["B"]))
    assert result.test == [1]
    with pytest.raises(ValueError, match="Unknown laboratory"):
        split(df, SplitConfig("laboratory", ["lab"], holdout_values=["D"]))


def test_laboratory_group_conflict_is_error():
    df = pd.DataFrame({"lab": ["A", "B", "C"], "sample": ["x", "x", "y"]})
    with pytest.raises(ValueError, match="connect held-out"):
        split(df, SplitConfig("laboratory", ["lab"], group_columns=["sample"], holdout_values=["B"]))


def test_time_uses_utc_and_keeps_cutoff_in_training():
    df = pd.DataFrame({"time": ["2025-01-01T00:00:00Z", "2024-12-31T19:00:00-05:00", "2025-01-02"]})
    result = split(df, SplitConfig("time", ["time"], cutoff="2025-01-01"))
    assert result.train == [0, 1]
    assert result.test == [2]
    assert_partition(result, 3)


@pytest.mark.parametrize("strategy,values,extra", [
    ("time", ["2025-01-01", "2025-01-03", "2025-01-01", "2025-01-03"], {"cutoff": "2025-01-02"}),
    ("extrapolation", [0, 3, 0, 3], {"threshold": 2}),
])
def test_crossing_groups_rejected_or_entirely_excluded(strategy, values, extra):
    df = pd.DataFrame({"x": values, "cell": ["a", "a", "b", "c"]})
    with pytest.raises(ValueError, match="cross the split boundary"):
        split(df, SplitConfig(strategy, ["x"], group_columns=["cell"], **extra))
    result = split(df, SplitConfig(strategy, ["x"], group_columns=["cell"], crossing_policy="exclude", **extra))
    assert result.excluded == [0, 1]
    assert result.train == [2] and result.test == [3]
    assert_partition(result, 4)


@pytest.mark.parametrize("direction,expected", [("high", [3]), ("low", [0, 1])])
def test_extrapolation_strict_boundary(direction, expected):
    df = pd.DataFrame({"x": [0, 1, 2, 3]})
    result = split(df, SplitConfig("extrapolation", ["x"], threshold=2, direction=direction))
    assert result.test == expected
    assert 2 in result.train


def test_actual_rdkit_scaffolds_and_acyclic_group():
    pytest.importorskip("rdkit")
    df = pd.DataFrame({"smiles": ["c1ccccc1", "Cc1ccccc1", "c1ccncc1", "CC", "CCC", "C1CCCCC1"]})
    result = split(df, SplitConfig("scaffold", ["smiles"], test_size=0.5))
    labels = result.assignments()
    assert labels[0] == labels[1]
    assert labels[3] == labels[4]
    assert result.diagnostics["scientific_groups"] == 4
    assert_partition(result, len(df))


@pytest.mark.parametrize("smiles", ["not-a-molecule", "[Na+].[Cl-]"])
def test_invalid_or_disconnected_smiles_rejected(smiles):
    pytest.importorskip("rdkit")
    with pytest.raises(ValueError, match="SMILES"):
        split(pd.DataFrame({"s": ["c1ccccc1", smiles]}), SplitConfig("scaffold", ["s"]))


@pytest.mark.parametrize("strategy,values,extra", [
    ("composition", ["a", "a"], {}), ("laboratory", ["a", None], {}),
    ("time", ["invalid", "2025-01-01"], {"cutoff": "2025-01-01"}),
    ("time", [1, 2], {"cutoff": "2025-01-01"}),
    ("time", [1, "2025-01-01"], {"cutoff": "2025-01-01"}),
    ("time", ["today", "2025-01-01"], {"cutoff": "2025-01-01"}),
    ("extrapolation", [0, float("inf")], {"threshold": 2}),
    ("extrapolation", [0, 1], {"threshold": 2}),
])
def test_unsafe_or_empty_splits_rejected(strategy, values, extra):
    with pytest.raises(ValueError):
        split(pd.DataFrame({"x": values}), SplitConfig(strategy, ["x"], **extra))


@pytest.mark.parametrize("kwargs", [
    {"test_size": 0}, {"test_size": 1}, {"seed": True}, {"threshold": 2}, {"group_columns": "cell"},
    {"holdout_values": ["a"]}, {"crossing_policy": "exclude"},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SplitConfig("composition", ["x"], **kwargs)


def test_fingerprint_changes_with_row_order_and_values():
    df = pd.DataFrame({"x": [1, 2, 3]})
    cfg = SplitConfig("composition", ["x"])
    assert split(df, cfg).metadata["dataset_sha256"] != split(df.iloc[::-1], cfg).metadata["dataset_sha256"]


def test_relative_cutoff_rejected():
    with pytest.raises(ValueError, match="absolute ISO"):
        SplitConfig("time", ["x"], cutoff="today")


def test_missing_rdkit_has_actionable_error(monkeypatch):
    import builtins
    original = builtins.__import__

    def without_rdkit(name, *args, **kwargs):
        if name == "rdkit" or name.startswith("rdkit."):
            raise ImportError("RDKit intentionally unavailable in this test")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_rdkit)
    with pytest.raises(ValueError, match="pip install"):
        split(pd.DataFrame({"s": ["CC", "c1ccccc1"]}), SplitConfig("scaffold", ["s"]))
