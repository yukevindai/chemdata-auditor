"""CSV input, JSON configuration, and explicit output paths."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from .audit import AuditConfig, audit
from .split import SplitConfig, split


def _csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    if not header or any(not c.strip() for c in header) or len(set(header)) != len(header):
        raise ValueError("CSV headers must be nonempty and unique.")
    # Preserve leading-zero IDs and literal identifiers like 'NA'. Numeric checks
    # convert only configured measurement columns; blank cells stay explicit.
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit scientific CSV datasets and design explicit holdouts.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "split"):
        command = commands.add_parser(name)
        command.add_argument("input", type=Path)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True, help="New JSON report path; existing files are never overwritten")
        if name == "audit":
            command.add_argument("--fail-on", choices=("error", "warning", "never"), default="never")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")
        data = _csv(args.input)
        raw = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a JSON object.")
        if args.command == "audit":
            result = audit(data, AuditConfig(**raw))
        else:
            result = split(data, SplitConfig(**raw))
        result.metadata["input_file_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
        payload = result.to_json()
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
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def split_main():
    return main(["split", *sys.argv[1:]])
