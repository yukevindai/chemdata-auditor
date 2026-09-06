"""CSV input, JSON configuration, and explicit output paths."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import pandas as pd

from .audit import AuditConfig, audit
from .split import SplitConfig, split
from .reporting import render
from .plugins import available_plugins, preset


def _csv_snapshot(path):
    snapshot = path.read_bytes()
    text = snapshot.decode("utf-8-sig")
    rows = csv.reader(io.StringIO(text), strict=True)
    header = next(rows, [])
    if not header or any(not c.strip() for c in header) or len(set(header)) != len(header):
        raise ValueError("CSV headers must be nonempty and unique.")
    for line, row in enumerate(rows, start=2):
        if row and len(row) != len(header):
            raise ValueError(f"CSV record {line} has {len(row)} fields; expected {len(header)}.")
    data = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    return data, hashlib.sha256(snapshot).hexdigest()


def _csv(path):
    return _csv_snapshot(path)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit scientific CSV datasets and design explicit holdouts.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plugins", help="List optional domain presets without loading external code")
    domain = commands.add_parser("preset", help="Generate an explicit audit configuration from a domain preset")
    domain.add_argument("name")
    domain.add_argument("--columns", required=True, type=Path, help="JSON role-to-column mapping")
    domain.add_argument("--output", required=True, type=Path)
    for name in ("audit", "split"):
        command = commands.add_parser(name)
        command.add_argument("input", type=Path)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True, help="New JSON report path; existing files are never overwritten")
        command.add_argument("--format", choices=("json", "markdown", "html"), default="json")
        if name == "audit":
            command.add_argument("--fail-on", choices=("error", "warning", "never"), default="never")
    args = parser.parse_args(argv)
    try:
        if args.command == "plugins":
            print(json.dumps(available_plugins(), indent=2))
            return 0
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")
        if args.command == "preset":
            from dataclasses import asdict
            cfg = preset(args.name, json.loads(args.columns.read_text(encoding="utf-8")))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(cfg), indent=2) + "\n")
            return 0
        data, digest = _csv_snapshot(args.input)
        raw = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a JSON object.")
        if args.command == "audit":
            result = audit(data, AuditConfig(**raw))
        else:
            result = split(data, SplitConfig(**raw))
        result.metadata["input_file_sha256"] = digest
        payload = render(result, args.format)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        print(f"Wrote {args.command} report to {args.output}")
        if args.command == "audit":
            print(f"{len(result.findings)} findings across {result.n_rows} rows")
            levels = {"error"} if args.fail_on == "error" else {"warning", "error"}
            if args.fail_on != "never" and any(f.severity in levels for f in result.findings):
                return 1
        return 0
    except (OSError, ValueError, TypeError, csv.Error, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def split_main():
    return main(["split", *sys.argv[1:]])
