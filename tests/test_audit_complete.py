import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from chemdata_auditor import AuditConfig, audit
from chemdata_auditor.cli import main
from chemdata_auditor.plugins import available_plugins, preset
from chemdata_auditor.reporting import render


def found(report, code):
    return [f for f in report.findings if f.code == code]


def test_missing_constant_conflicting_identifier_and_target_proxy():
    df = pd.DataFrame({"id": ["1", "1", "2", "3", "4", "5"], "row_id": list(range(6)),
                       "y": [1, 2, 3, 4, 5, 6], "proxy": [3, 5, 7, 9, 11, 13],
                       "constant": [1] * 6, "source": [None, "unknown", "bad", "10.1/a", "10.1/b", "10.1/c"]})
    report = audit(df, AuditConfig(identity_columns=["id"], conflict_columns=["y"], feature_columns=["row_id", "proxy"],
                    target_column="y", provenance_columns=["source"], provenance_patterns={"source": r"10\.\d+/.+"}))
    assert found(report, "conflicting_samples")[0].rows == [0, 1]
    assert found(report, "suspicious_identifier")[0].columns == ["row_id"]
    assert found(report, "target_proxy")
    assert found(report, "weak_provenance")[0].rows == [1, 2]
    assert found(report, "missing_values")[0].rows == [0]
    for finding in report.findings:
        assert "raw_examples" in finding.details and finding.suggestion
    json.loads(report.to_json())


def test_unit_conversion_precedes_bounds_without_mutation():
    df = pd.DataFrame({"t": [-10, 298, 50, 3], "u": ["degC", "K", "m", "mystery"]})
    original = df.copy(deep=True)
    report = audit(df, AuditConfig(units={"t": {"column": "u", "expected": "K"}}, bounds={"t": [0, 400]}))
    assert not found(report, "out_of_bounds")
    assert found(report, "unit_conversion")[0].details["converted_examples"]["0"] == pytest.approx(263.15)
    assert found(report, "incompatible_units")[0].rows == [2]
    assert found(report, "unknown_unit")[0].rows == [3]
    pd.testing.assert_frame_equal(df, original)


def test_unit_aliases_and_concentration_conversion():
    df = pd.DataFrame({"c": [1000, 2], "u": ["millimolar", "mol/L"]})
    report = audit(df, AuditConfig(units={"c": {"column": "u", "expected": "mol/L"}},
                    unit_aliases={"millimolar": "mmol/L"}, bounds={"c": [0, 1.5]}))
    assert found(report, "out_of_bounds")[0].rows == [1]


def test_composition_and_safe_conditional_chronological_rules():
    df = pd.DataFrame({"a": [50, 60, -1, None], "b": [50, 30, 101, 10],
        "kind": ["salt", "salt", "other", "salt"], "c": [1, -1, -1, None],
        "start": ["2025-01-01"] * 4, "end": ["2025-01-02", "2024-01-01", "bad", "2025-01-03"]})
    cfg = AuditConfig(compositions=[{"columns": ["a", "b"], "total": 100}], rules=[
        {"name": "Positive salt concentration", "when": {"column": "kind", "op": "eq", "value": "salt", "kind": "string"},
         "assert": {"column": "c", "op": "ge", "value": 0}},
        {"name": "Measurements follow preparation", "assert": {"column": "start", "op": "on_or_before", "other_column": "end"}},
    ])
    report = audit(df, cfg)
    assert found(report, "composition_error")[0].rows == [1, 2]
    assert found(report, "composition_not_evaluable")[0].rows == [3]
    assert [f.rows for f in found(report, "constraint_violation")] == [[1], [1]]
    assert [f.rows for f in found(report, "constraint_not_evaluable")] == [[3], [2]]


def test_three_valued_logic_false_conjunction_does_not_invent_failure():
    df = pd.DataFrame({"a": [0, 1], "b": [None, None], "c": [-1, -1]})
    cfg = AuditConfig(rules=[{"name": "conditional", "when": {"all": [
        {"column": "a", "op": "eq", "value": 1}, {"column": "b", "op": "eq", "value": 1}]},
        "assert": {"column": "c", "op": "ge", "value": 0}}])
    report = audit(df, cfg)
    assert not found(report, "constraint_violation")
    assert found(report, "constraint_not_evaluable")[0].rows == [1]


def test_numeric_near_duplicates_overlap_and_density():
    df = pd.DataFrame({"x": [0, 0.01, 10, None], "y": [0, 0.01, 10, 2], "part": ["train", "test", "test", "train"]})
    cfg = AuditConfig(near_duplicates={"x": 0.02, "y": 0.02}, split_column="part",
                      density_columns=["x", "y"], density_scales={"x": 1, "y": 1}, min_neighbors=1)
    report = audit(df, cfg)
    assert found(report, "near_duplicate_split_overlap")[0].rows == [0, 1]
    assert found(report, "sparse_neighborhood")[0].rows == [2]
    assert found(report, "similarity_not_evaluable")[0].rows == [3]
    assert found(report, "density_not_evaluable")[0].rows == [3]


def test_molecular_canonical_duplicates_similarity_and_invalid_structures():
    pytest.importorskip("rdkit")
    df = pd.DataFrame({"s": ["CCO", "OCC", "CCCO", "bad"], "part": ["train", "test", "test", "train"]})
    report = audit(df, AuditConfig(smiles_column="s", molecular_similarity=1, split_column="part"))
    assert found(report, "duplicate_molecules")[0].rows == [0, 1]
    assert found(report, "near_duplicate_split_overlap")[0].rows == [0, 1]
    assert found(report, "invalid_molecule")[0].rows == [3]


def test_report_formats_escape_user_supplied_html():
    df = pd.DataFrame({"x": [-1]})
    report = audit(df, AuditConfig(rules=[{"name": "<script>alert(1)</script>",
                                         "assert": {"column": "x", "op": "ge", "value": 0}}]))
    for fmt in ("html", "markdown"):
        result = render(report, fmt)
        assert "<script>" not in result and "&lt;script&gt;" in result
        assert "configuration" in result.lower()


def test_report_handles_nonfinite_raw_evidence():
    report = audit(pd.DataFrame({"x": [np.inf, np.nan, -np.inf]}), AuditConfig(numeric_columns=["x"]))
    assert "invalid_numeric" in report.to_json()
    json.loads(report.to_json())


@pytest.mark.parametrize("name,roles,bad", [
    ("batteries", {"capacity": "x"}, -1), ("materials", {"porosity_fraction": "x"}, 1.1),
    ("process", {"pressure_absolute": "x"}, -1), ("molecular", {"molecular_weight": "x"}, -1),
])
def test_domain_presets_are_executable(name, roles, bad):
    assert found(audit(pd.DataFrame({"x": [bad, 1]}), preset(name, roles)), "out_of_bounds")
    assert name in available_plugins()["builtin"]


def test_pair_budget_is_explicit_not_silently_sampled():
    with pytest.raises(ValueError, match="exceeding"):
        audit(pd.DataFrame({"x": [1, 2, 3]}), AuditConfig(near_duplicates={"x": 1}, max_pair_comparisons=1))


@pytest.mark.parametrize("kwargs", [
    {"near_duplicates": {"x": -1}}, {"density_columns": ["x"]}, {"max_pair_comparisons": 0},
    {"compositions": [{"columns": ["x"], "total": 0}]}, {"conflict_columns": ["x"]},
    {"rules": [{"name": "bad", "assert": {"column": "x", "op": "eval", "value": "__import__('os')"}}]},
    {"rules": [{"name": "bad", "assert": {"column": "x", "op": "ge", "value": 1, "other_column": "y"}}]},
    {"molecular_similarity": 0.5}, {"provenance_patterns": {"source": "["}},
])
def test_invalid_audit_config(kwargs):
    with pytest.raises(ValueError):
        AuditConfig(**kwargs)


def test_csv_snapshot_hash_matches_parsed_bytes_even_when_file_changes(tmp_path, monkeypatch):
    import chemdata_auditor.cli as cli
    source, cfg, output = [tmp_path / p for p in ("data.csv", "cfg.json", "report.json")]
    before = b"x\n1\n2\n"
    source.write_bytes(before)
    cfg.write_text("{}")
    original = cli.audit
    def concurrent_edit(data, config):
        source.write_text("x\n999\n")
        return original(data, config)
    monkeypatch.setattr(cli, "audit", concurrent_edit)
    assert main(["audit", str(source), "--config", str(cfg), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["n_rows"] == 2
    assert report["metadata"]["input_file_sha256"] == hashlib.sha256(before).hexdigest()


@pytest.mark.parametrize("contents", ["x,y\n1,2,3\n", "x,y\n1\n", 'x,y\n1,"oops\n'])
def test_ragged_or_malformed_csv_rejected(tmp_path, contents):
    source, cfg, output = [tmp_path / p for p in ("data.csv", "cfg.json", "report.json")]
    source.write_text(contents)
    cfg.write_text("{}")
    assert main(["audit", str(source), "--config", str(cfg), "--output", str(output)]) == 2
    assert not output.exists()


def test_preset_cli(tmp_path):
    roles, output = tmp_path / "roles.json", tmp_path / "audit.json"
    roles.write_text('{"capacity": "q"}')
    assert main(["preset", "batteries", "--columns", str(roles), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["bounds"] == {"q": [0, None]}
