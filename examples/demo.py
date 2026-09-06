"""Run all six example configurations and re-audit the composition split."""

import json
from pathlib import Path

import pandas as pd

from chemdata_auditor import AuditConfig, SplitConfig, audit, split

HERE = Path(__file__).resolve().parent
data = pd.read_csv(HERE / "electrolytes.csv", dtype=str, keep_default_na=False)
audit_cfg = json.loads((HERE / "audit.json").read_text())
report = audit(data, AuditConfig(**audit_cfg))
print("Original audit:", ", ".join(sorted({f.code for f in report.findings})))

for strategy in ("composition", "laboratory", "scaffold", "time", "extrapolation"):
    cfg = SplitConfig(**json.loads((HERE / f"{strategy}.json").read_text()))
    result = split(data, cfg)
    print(f"{strategy}: train={len(result.train)} test={len(result.test)} excluded={len(result.excluded)}")
    if strategy == "composition":
        assigned = data.copy()
        assigned["existing_split"] = result.assignments()
        followup = audit(assigned, AuditConfig(**audit_cfg))
        leakage = [f for f in followup.findings if f.code in {"duplicate_split_overlap", "group_split_overlap"}]
        assert not leakage, "The example composition split should resolve the original partition overlap."
        print("Composition re-audit: no duplicate or sample-group overlap; data quality findings remain.")
