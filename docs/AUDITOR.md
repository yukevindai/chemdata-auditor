# ChemData Auditor: checks, constraints, and domain plugins

The audit engine answers whether configured data and evaluation assumptions have evidence behind them. It returns findings, not a universal reliability score. Strong correlations and repeated molecules can be legitimate; investigate context before changing data.

## Run locally

```bash
python -m pip install -e '.[dev,chem]'
python -m chemdata_auditor audit examples/electrolytes.csv --config examples/audit-complete.json --output outputs/complete-audit.json
python -m chemdata_auditor audit examples/electrolytes.csv --config examples/audit-complete.json --output outputs/complete-audit.html --format html
python -m chemdata_auditor plugins
python -m pytest -q
```

Use `--format markdown` for a human-readable Markdown report. HTML reports are standalone, escape supplied text, and load no scripts or remote assets. JSON contains the same complete evidence and reproducible configuration. Evidence can contain input data: choose report sharing accordingly.

The CLI reads the CSV bytes once, validates unique headers and consistent record lengths, then parses and hashes that exact snapshot. Leading-zero identifiers and literal `NA` remain strings. Blank/null cells are missing; sentinel strings should be normalized explicitly when they mean missing measurements.

## Configuration reference

`AuditConfig(**config)` accepts the following JSON-compatible fields. Unknown fields and invalid configurations fail before a report is returned. Defaults never invent scientific bounds, units, or sample relationships.

| Fields | Behavior and defaults |
| --- | --- |
| `check_missing` | `true`: report null/blank cells in every column and their fractions. |
| `numeric_columns` | `[]`: explicitly validate measurements for missing, nonnumeric, and nonfinite values. Bounds/bin columns are also validated. |
| `duplicate_columns` | Default: all non-partition columns. Exact duplicate comparison; nulls compare equal. Specify scientifically meaningful observation keys for cross-source records. |
| `identity_columns`, `conflict_columns` | `[]`: joint identity of one observation and the fields that must agree for that identity. Include time/cycle/condition keys when those distinguish observations. |
| `conflict_tolerance` | `0`: absolute numeric tolerance in internally normalized units; nonnumeric disagreements use exact comparison. Missing outcomes do not establish a conflict. |
| `identifier_columns` | `[]`: identifiers to screen; features with names such as `row_id` are also screened. At least three observed values and >=90% uniqueness trigger a warning. |
| `feature_columns`, `target_column`, `unavailable_features` | Declared model inputs, optional outcome, and features unavailable when predictions are made. Explicit outcome/unavailable inputs are errors; exact copies and high correlations are warnings. |
| `correlation_threshold` | `0.995`: absolute Pearson correlation screening with at least five jointly observed nonconstant numeric values. Not proof of causal leakage. |
| `split_column`, `group_columns` | Optional partition labels and independently defined grouping columns. Check duplicate/group overlap across every distinct nonblank partition. Excluded rows should be removed before re-auditing. |
| `bounds` | `{}`: column to inclusive `[lower, upper]`. One endpoint may be null. Configure concentration ranges, nonnegative measurements, and physical constraints explicitly. |
| `units` | `{}`: measurement column to `{"column": "unit_label", "expected": "K"}`. Pint checks dimensions and converts a working copy to canonical units before bounds, numeric similarity, density, composition, and rule checks. |
| `unit_aliases` | `{}`: explicit mappings such as `{"C": "degC"}`. No guessing: Pint's `C` means coulomb, so Celsius should be `degC` unless mapped. |
| `compositions` | `[]`: objects with `columns`, positive `total` (default 100), and absolute `tolerance` (default 1e-6). Check nonnegative components and the sum. Missing components produce a not-evaluable finding. |
| `rules` | `[]`: declarative conditional and relational constraints described below. |
| `near_duplicates` | `{}`: column to nonnegative absolute tolerance. Two rows match if every configured numeric difference is within tolerance. Invalid rows are reported as not evaluable. |
| `smiles_column` | Optional SMILES column; RDKit checks validity and canonical isomeric molecular identity. Invalid/disconnected structures are reported without guessing salt handling. |
| `molecular_similarity` | Optional Tanimoto threshold in (0,1], requiring `smiles_column`. Morgan radius 2, 2048 bits, chirality enabled. Fingerprint similarity is a heuristic, not chemical equivalence. |
| `sparse_bins`, `min_bin_count` | Column to increasing numeric edges; minimum row count defaults to 3. Last bin includes its right edge. Empty intervals and out-of-range observations are reported. |
| `density_columns`, `density_scales` | Multivariate numeric columns and an explicit positive scale per column. No scaling is fitted to data. |
| `density_radius`, `min_neighbors` | Radius defaults to 1 in scaled Euclidean space; minimum neighbors defaults to 3. Self is excluded, while repeated measurements still count as rows. |
| `max_pair_comparisons` | 2,000,000 per numeric/molecular/density calculation. Above-budget work fails explicitly; no silent sampling. Algorithms use bounded row-wise arrays rather than a full distance matrix. |
| `provenance_columns` | Metadata whose presence is required. Missing columns/cells and placeholders such as `unknown` or `TODO` are findings. |
| `provenance_patterns` | Optional full-match regex per source column, e.g. a DOI syntax rule. Patterns are trusted local configuration, not a source authentication mechanism. |

Constant columns are reported as informational schema findings. Every finding includes a code, severity, all affected row positions, affected columns, a message, recommended action, and evidence. Raw examples are capped at 20 rows and similarity pair examples at 100; truncation is explicit and counts/affected rows are complete. Strict JSON replaces missing numeric evidence with null and represents infinities as strings.

Unit-incompatible, unknown, or missing measurements are not used as valid normalized numbers in physical checks. Input data is never changed. Bound values, numeric rule thresholds, composition totals, and similarity tolerances must use the configured canonical units. Provenance, exact record identity, identifier screening, and target proxy screening inspect the original data.

## Safe rule language

Rules contain `name`, `assert`, and optional `when`, `severity` (`error` by default), and `action`. Predicates contain `column`, `op`, and exactly one of `value` or `other_column`. Default `kind` is `numeric`; use `string` for categories. There is no executable expression language or `eval`.

```json
{
  "rules": [
    {
      "name": "Salt concentration is positive when salt is present",
      "when": {"column": "has_salt", "op": "eq", "value": "yes", "kind": "string"},
      "assert": {"column": "salt_concentration", "op": "gt", "value": 0},
      "action": "Check salt amount, solution volume, and reported concentration."
    },
    {
      "name": "Measurement follows preparation",
      "assert": {"column": "prepared_at", "op": "on_or_before", "other_column": "measured_at"}
    }
  ]
}
```

Comparison operators are `eq`, `ne`, `lt`, `le`, `gt`, `ge`; membership operators are `in` and `not_in` with a nonempty value list. Chronological operators are `before`, `after`, `on_or_before`, `on_or_after` with absolute ISO dates/timestamps (naive dates interpreted as UTC). Combine predicates with `{"all": [...]}` or `{"any": [...]}` up to eight levels. Missing or invalid inputs yield `constraint_not_evaluable`, using three-valued logic; they cannot silently pass as valid evidence.

## Optional domain plugins

Built-in presets activate only when explicitly selected. They map semantic roles to your actual columns and return a normal, editable `AuditConfig`:

```python
from chemdata_auditor import audit
from chemdata_auditor.plugins import preset

config = preset("batteries", {"capacity": "q_mAh", "cycle": "cycle_index", "source": "doi"})
report = audit(df, config)
```

| Preset | Supported measurement roles |
| --- | --- |
| `batteries` | `temperature_k`, `capacity`, `cycle`, `salt_concentration` |
| `molecular` | `molecular_weight`, `smiles` (requires optional RDKit) |
| `materials` | `density`, `porosity_fraction`, `temperature_k` |
| `process` | `pressure_absolute`, `temperature_k`, `mass_flow_magnitude` |

Every preset also supports `sample_id`, `source`, `laboratory`, and `measured_at`. These roles have precise meanings: absolute pressure is not gauge pressure; a flow magnitude is not signed directional flow. Review presets for your experimental convention. Add application-specific upper bounds and compositions as normal configuration; there are no universal concentration limits.

The CLI can generate a preset config from a role mapping JSON: `chemdata preset batteries --columns roles.json --output audit.json`. External Python packages can register a callable under the `chemdata_auditor.presets` entry-point group; it receives the role mapping and must return `AuditConfig`. Listing plugins does not load their code. Explicitly selecting an installed external plugin executes that trusted local plugin.

## Interpretation and validation

Tests cover unit conversion with offsets, incompatible dimensions, conflict identity, missing evidence, conditional/chronological rules, composition errors, numeric/molecular near duplicates, joint density, strict CSV snapshots, rendering escaping, and all four presets. These tests validate implemented behavior; they do not establish accuracy on all scientific datasets.

This toolkit cannot infer all hidden confounders, certify causal generalization, verify the authenticity of a publication, or know whether a local ID is globally unique. Reports make assumptions and uncovered areas visible so researchers can judge the claim they intend to make.

Implementation references: [Pint unit conversion](https://pint.readthedocs.io/en/stable/getting/tutorial.html), [Pint offset temperatures](https://pint.readthedocs.io/en/stable/user/nonmult.html), and [RDKit fingerprint generators](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html).
