import json

from chemdata_auditor.cli import main


def test_cli_audit_exit_codes_and_no_overwrite(tmp_path):
    source, config, output = [tmp_path / x for x in ("data.csv", "config.json", "report.json")]
    source.write_text("x\n-1\n2\n")
    config.write_text(json.dumps({"bounds": {"x": [0, None]}}))
    args = ["audit", str(source), "--config", str(config), "--output", str(output), "--fail-on", "error"]
    assert main(args) == 1
    report = json.loads(output.read_text())
    assert report["findings"][0]["code"] == "out_of_bounds"
    assert len(report["metadata"]["input_file_sha256"]) == 64
    original = output.read_bytes()
    assert main(args) == 2
    assert output.read_bytes() == original


def test_cli_preserves_leading_zero_and_na_identifiers(tmp_path):
    source, config, output = [tmp_path / x for x in ("data.csv", "config.json", "report.json")]
    source.write_text("lab\n001\n1\nNA\n")
    config.write_text(json.dumps({"strategy": "laboratory", "columns": ["lab"], "holdout_values": ["001"]}))
    assert main(["split", str(source), "--config", str(config), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["test"] == [0]


def test_cli_invalid_config_has_no_output(tmp_path):
    source, config, output = [tmp_path / x for x in ("data.csv", "config.json", "report.json")]
    source.write_text("x\n1\n2\n")
    config.write_text('{"stratgey": "composition"}')
    assert main(["split", str(source), "--config", str(config), "--output", str(output)]) == 2
    assert not output.exists()


def test_duplicate_csv_headers_rejected(tmp_path):
    source, config, output = [tmp_path / x for x in ("data.csv", "config.json", "report.json")]
    source.write_text("x,x\n1,2\n")
    config.write_text('{}')
    assert main(["audit", str(source), "--config", str(config), "--output", str(output)]) == 2
