import json

import numpy as np
import pandas as pd
import pytest

from chemdata_auditor import DiagnosticsConfig, SplitConfig, SplitResult, compare_splits, diagnose, split
from chemdata_auditor.cli import main
from chemdata_auditor.reporting import render


def disjoint(result, n, validation=True):
    parts = [result.train, result.validation, result.test, result.excluded]
    rows = [i for p in parts for i in p]
    assert len(rows) == n and set(rows) == set(range(n))
    assert result.train and result.test and (result.validation or not validation)
    assert len(result.assignments()) == n
    assert result.diagnostics["generalization"]["supports"]
    assert result.findings[-1].suggestion
    json.loads(result.to_json())


@pytest.mark.parametrize("strategy", ["random", "formulation", "composition", "publication", "laboratory"])
def test_seeded_three_way_strategies(strategy):
    df = pd.DataFrame({"key": list("aabbccddeeff"), "sample": list("aabbccddeeff"), "x": list(range(12))}, index=[100] * 12)
    cfg = SplitConfig(strategy, [] if strategy == "random" else ["key"], group_columns=["sample"],
                      test_size=0.25, validation_size=0.25, seed=9)
    result = split(df, cfg)
    disjoint(result, 12)
    labels = result.assignments()
    assert all(labels[i] == labels[i + 1] for i in range(0, 12, 2))
    assert split(df, cfg) == result


@pytest.mark.parametrize("strategy", ["publication", "formulation", "laboratory"])
def test_explicit_three_way_identities(strategy):
    df = pd.DataFrame({"key": ["A", "B", "C", "D"]})
    result = split(df, SplitConfig(strategy, ["key"], holdout_values=["D"], validation_values=["C"]))
    assert result.train == [0, 1] and result.validation == [2] and result.test == [3]
    disjoint(result, 4)


def test_composition_normalization_and_declared_rounding():
    df = pd.DataFrame({"x": ["0.2", "0.20", "0.20001", "0.4", "0.6", "0.8"]})
    result = split(df, SplitConfig("composition", ["x"], composition_decimals=3, validation_size=0.2))
    assert result.assignments()[0] == result.assignments()[1] == result.assignments()[2]
    assert result.diagnostics["scientific_groups"] == 4


def test_three_way_scaffold_and_clusters():
    pytest.importorskip("rdkit")
    df = pd.DataFrame({"s": ["c1ccccc1", "Cc1ccccc1", "c1ccncc1", "C1CCCCC1", "CC", "CCC"]})
    result = split(df, SplitConfig("scaffold", ["s"], validation_size=0.25, test_size=0.25))
    disjoint(result, 6)
    assert result.assignments()[0] == result.assignments()[1]
    values = pd.DataFrame({"x": [0, 0.1, 10, 10.1, 20, 20.1], "y": [0, 0.1, 10, 10.1, 20, 20.1]})
    cfg = SplitConfig("cluster", ["x", "y"], n_clusters=3, cluster_scales={"x": 1, "y": 1}, validation_size=0.2)
    clustered = split(values, cfg)
    disjoint(clustered, 6)
    assert all(clustered.assignments()[i] == clustered.assignments()[i + 1] for i in [0, 2, 4])
    assert clustered == split(values, cfg)


def test_temporal_three_way_cutoff_equality_and_timezones():
    df = pd.DataFrame({"t": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"]})
    result = split(df, SplitConfig("temporal", ["t"], validation_cutoff="2025-01-31T19:00:00-05:00", cutoff="2025-03-01"))
    assert result.train == [0, 1] and result.validation == [2] and result.test == [3]
    disjoint(result, 4)


@pytest.mark.parametrize("direction,threshold,val_threshold,expected", [
    ("high", 3, 1, ([0, 1], [2, 3], [4, 5])),
    ("low", 2, 4, ([4, 5], [2, 3], [0, 1])),
])
def test_three_way_threshold_extrapolation(direction, threshold, val_threshold, expected):
    result = split(pd.DataFrame({"x": range(6)}), SplitConfig("extrapolation", ["x"], threshold=threshold,
                   validation_threshold=val_threshold, direction=direction))
    assert (result.train, result.validation, result.test) == expected


def test_multivariate_box_extrapolation():
    df = pd.DataFrame({"x": [0, 1, 2, 3, 0], "y": [0, 1, 0, 0, 4]})
    cfg = SplitConfig("extrapolation", ["x", "y"], extrapolation_bounds={"x": [0, 2], "y": [0, 2]},
                      validation_bounds={"x": [0, 1], "y": [0, 1]})
    result = split(df, cfg)
    assert result.train == [0, 1] and result.validation == [2] and result.test == [3, 4]
    disjoint(result, 5)


def test_similarity_grouping_uses_transitive_components():
    df = pd.DataFrame({"x": [0, 0.09, 0.18, 1, 2, 3]})
    cfg = SplitConfig("random", validation_size=0.2, group_near_duplicates={"x": 0.1}, diagnostics={"near_duplicates": {"x": 0.1}})
    result = split(df, cfg)
    assert len(set(result.assignments()[:3])) == 1
    assert not any(f.code == "near_duplicate_overlap" for f in result.findings)
    disjoint(result, 6)


def test_molecular_grouping_prevents_equivalent_smiles_leakage():
    pytest.importorskip("rdkit")
    df = pd.DataFrame({"s": ["CCO", "OCC", "c1ccccc1", "c1ccncc1", "C1CCCCC1"]})
    cfg = SplitConfig("random", validation_size=0.2, group_smiles_column="s", group_molecular_similarity=1,
                      diagnostics={"smiles_column": "s", "molecular_similarity": 1})
    result = split(df, cfg)
    assert result.assignments()[0] == result.assignments()[1]
    assert not any(f.code == "near_duplicate_overlap" for f in result.findings)


def test_boundary_exclusion_covers_train_validation_and_test():
    df = pd.DataFrame({"x": [0, 2, 4, 0, 2, 4], "g": ["cross", "cross", "cross", "a", "b", "c"]})
    cfg = SplitConfig("extrapolation", ["x"], threshold=3, validation_threshold=1, group_columns=["g"], crossing_policy="exclude")
    result = split(df, cfg)
    assert result.excluded == [0, 1, 2]
    assert result.train == [3] and result.validation == [4] and result.test == [5]
    disjoint(result, 6)


def test_diagnostics_find_validation_leakage_and_known_shifts():
    df = pd.DataFrame({"id": ["a", "b", "a", "c", "d", "e"], "x": [0, 1, 0, 10, 11, 12], "cat": ["A", "A", "A", "B", "B", "B"]})
    result = SplitResult([0, 1], [4, 5], [], {}, {}, validation=[2, 3])
    metrics, findings = diagnose(df, result, DiagnosticsConfig(group_columns=["id"], duplicate_columns=["id", "x"],
                                 near_duplicates={"x": 0.01}, numeric_columns=["x"], categorical_columns=["cat"]))
    assert metrics["pairwise_overlap"]["train:validation"]["groups"]["id"] == 1
    assert metrics["pairwise_overlap"]["train:validation"]["exact_duplicate_pairs"] == 1
    assert metrics["numeric_near_duplicate_pairs"]["train:validation"] == 1
    assert metrics["distribution_shift"]["test"]["numeric"]["x"]["ks_statistic"] == 1
    assert metrics["distribution_shift"]["test"]["categorical"]["cat"]["total_variation"] == 1
    assert {"group_overlap", "exact_duplicate_overlap", "near_duplicate_overlap"} <= {f.code for f in findings}


def test_domain_fit_uses_training_only_and_detects_joint_hole():
    df = pd.DataFrame({"x": [0, 1, 2, 0, 100], "y": [0, 1, 2, 2, 100]})
    result = SplitResult([0, 1, 2], [3, 4], [], {}, {})
    cfg = DiagnosticsConfig(domain_columns=["x", "y"], domain_distance_threshold=0.1)
    metrics, findings = diagnose(df, result, cfg)
    domain = metrics["training_domain"]
    assert domain["fit_rows"] == [0, 1, 2]
    assert domain["scales"] == pytest.approx([np.std([0, 1, 2])] * 2)
    assert domain["rows"][0]["outside_training_range"] is False
    assert domain["rows"][0]["beyond_distance_threshold"] is True
    assert domain["outside_rows"] == [3, 4]
    df.loc[4, ["x", "y"]] = [10000, -10000]
    second, _ = diagnose(df, result, cfg)
    assert second["training_domain"]["scales"] == domain["scales"]
    assert second["training_domain"]["distance_threshold"] == domain["distance_threshold"]


def test_domain_calibration_ignores_test_extremes_and_handles_constant_train():
    df = pd.DataFrame({"x": [1, 1, 2]})
    result = SplitResult([0, 1], [2], [], {}, {})
    metrics, _ = diagnose(df, result, DiagnosticsConfig(domain_columns=["x"]))
    assert metrics["training_domain"]["distance_threshold"] == 0
    assert metrics["training_domain"]["outside_rows"] == [2]
    assert metrics["training_domain"]["constant_training_dimensions"] == ["x"]


def test_snapshot_and_invalid_partition_validation():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    result = split(df, SplitConfig("random"))
    with pytest.raises(ValueError, match="snapshot"):
        diagnose(df.iloc[::-1], result)
    with pytest.raises(ValueError, match="exactly once"):
        diagnose(df, SplitResult([0, 1], [1, 2], [], {}, {}))


def test_comparison_records_infeasible_design_and_scope_without_fake_scores():
    df = pd.DataFrame({"x": [0, 1, 2, 3], "lab": ["A", "A", "B", "B"]})
    result = compare_splits(df, {"random": {"strategy": "random"},
                                "lab": {"strategy": "laboratory", "columns": ["lab"], "holdout_values": ["B"]},
                                "impossible": {"strategy": "time", "columns": ["x"]}}, DiagnosticsConfig(numeric_columns=["x"]))
    assert result.results["random"]["status"] == "ok"
    assert result.results["lab"]["summary"]["n_test"] == 2
    assert result.results["impossible"]["status"] == "error"
    assert "generalization" in result.results["lab"]["summary"]
    assert "SciSplit comparison" in render(result, "html")
    json.loads(result.to_json())


@pytest.mark.parametrize("kwargs", [
    {"strategy": "random", "validation_size": 0.9},
    {"strategy": "cluster", "columns": ["x"]},
    {"strategy": "time", "columns": ["t"], "cutoff": "2025-01-01", "validation_size": 0.2},
    {"strategy": "extrapolation", "columns": ["y"], "threshold": 1, "target_column": "y"},
    {"strategy": "laboratory", "columns": ["lab"], "holdout_values": ["A"], "validation_values": ["A"]},
    {"strategy": "extrapolation", "columns": ["x"], "extrapolation_bounds": {"x": [0, 1]}, "validation_bounds": {"x": [0, 2]}},
    {"strategy": "random", "group_near_duplicates": {"x": -1}},
    {"strategy": "random", "diagnostics": {"domain_scales": {"x": 0}}},
])
def test_invalid_designs_rejected(kwargs):
    with pytest.raises(ValueError):
        SplitConfig(**kwargs)


def test_cli_three_way_exports_comparison_and_rediagnosis(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("x,lab\n0,A\n1,B\n2,C\n3,D\n4,E\n5,F\n")
    cfg, report, assignments = [tmp_path / p for p in ("split.json", "report.json", "assignments.csv")]
    cfg.write_text(json.dumps({"strategy": "random", "validation_size": 0.2}))
    assert main(["split", str(source), "--config", str(cfg), "--output", str(report), "--assignments", str(assignments)]) == 0
    labels = pd.read_csv(assignments)
    assert set(labels.partition) == {"train", "validation", "test"}
    diagnostics, output = tmp_path / "diag.json", tmp_path / "recheck.html"
    diagnostics.write_text('{"numeric_columns": ["x"], "domain_columns": ["x"]}')
    assert main(["diagnose", str(source), "--config", str(diagnostics), "--split-report", str(report), "--output", str(output), "--format", "html"]) == 0
    compare, comparison = tmp_path / "compare.json", tmp_path / "comparison.json"
    compare.write_text('{"strategies": {"random": {"strategy": "random"}, "invalid": {"strategy": "bad"}}}')
    assert main(["compare", str(source), "--config", str(compare), "--output", str(comparison)]) == 1
    assert json.loads(comparison.read_text())["results"]["invalid"]["status"] == "error"
