# Configuration and scientific assumptions

Both commands accept a JSON object. Unknown fields cause an error. Python callers construct the equivalent `AuditConfig` or `SplitConfig`. These tools accept nonempty tables with unique string column names and scalar cell values. They never impute, overwrite source measurements, deduplicate, or train a model automatically.

## Audit configuration

The expanded audit configuration and domain plugins are documented in [AUDITOR.md](AUDITOR.md). Unit conversions are performed only in an internal copy; source data is preserved.

## Split configuration

`strategy` and `columns` are required. Relevant grouping and split metadata must be complete; missing or blank identifiers cause an error.

| Field | Default | Meaning |
| --- | --- | --- |
| `strategy` | Required | `composition`, `scaffold`, `laboratory`, `time`, or `extrapolation`. |
| `columns` | Required | One column, or multiple columns jointly defining a composition. |
| `group_columns` | `[]` | Columns identifying related samples that must stay together. Each column independently defines a grouping constraint. |
| `test_size` | `0.2` | Fraction of independent connected groups to select for seeded holdout. Must be strictly between 0 and 1. |
| `seed` | `0` | Integer seed for group shuffling. Reproducibility assumes the same data, row order, configuration, and software environment. |
| `holdout_values` | Unset | Explicit laboratory labels to hold out. Unknown labels and group conflicts cause errors. |
| `cutoff` | Unset | Required absolute ISO date/timestamp for time splits. Training includes the cutoff; test is strictly later. |
| `threshold` | Unset | Required finite threshold for extrapolation splits. Equality stays in training. |
| `direction` | `high` | Extrapolation: `high` tests values above the threshold; `low` tests values below it. |
| `crossing_policy` | `error` | Time/extrapolation only: `error`, or `exclude` to remove an entire group crossing the boundary. |

Nondefault `test_size` and `seed` are rejected for explicit laboratory holdouts and boundary splits because these settings cannot affect the result. Set them only for seeded group holdouts. Other strategy-specific fields are rejected when they do not apply.

### Grouping and allocation

For composition, scaffold, and laboratory strategies, rows sharing a scientific key or any declared sample-group value are connected. The transitive connected components are indivisible. For example, if rows 0 and 1 share a cell and rows 1 and 2 share a batch, all three rows stay together. Identifiers must therefore be globally meaningful within each grouping column; create a composite laboratory-plus-local-ID column if local IDs are reused.

Seeded allocation shuffles components and holds out `ceil(test_size * number_of_components)`, capped to leave at least one training component. This is a **group fraction**, not a guaranteed row fraction. Uneven group sizes can yield a very different test row fraction; the actual fraction is reported. At least two independent components are needed. Explicit laboratory holdouts do not silently expand to include unselected laboratories when grouping constraints conflict.

Composition keys use exact values, without rounding, unit normalization, composition-sum checks, or chemical equivalence inference. In the CSV interface, `0.2` and `0.20` remain different strings. Normalize representations upstream or supply a validated formulation-family column. Multiple numerical columns form a joint key; no distance metric is applied.

Scaffold splitting requires the optional `chem` extra. It uses RDKit Bemis–Murcko scaffold SMILES with chirality excluded. All acyclic compounds share one group. Invalid or disconnected SMILES raise an error; the implementation does not guess which salt fragment to retain. Scaffold identity is not a guarantee of full chemical dissimilarity. See the [RDKit API reference](https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html).

Time splits accept absolute ISO timestamps. Naive timestamps are interpreted as UTC, and timezone-aware values are normalized to UTC. Numeric timestamps are rejected because their units are ambiguous. Supply the date when information would actually have been available, not a later digitization or upload date unless that matches the evaluation question.

Extrapolation is deliberately one-dimensional and threshold-based. Choose the dimension and boundary before inspecting test performance. Using an outcome to define a holdout creates an outcome-conditioned benchmark and must be disclosed. It does not demonstrate generalization across all chemical dimensions.

For time and extrapolation, a declared group crossing the threshold triggers an error by default. `crossing_policy: "exclude"` excludes all its rows while preserving strict boundary separation. Exclusion can introduce selection bias and may leave an empty partition, which is an error. No rows are silently reassigned to relax the scientific boundary.

### Reviewing and using results

Use `df.iloc[result.train]` and `df.iloc[result.test]`, not index labels. `result.assignments()` returns `train`, `test`, or `excluded` for every row in input order. If re-auditing split assignments, remove explicitly excluded rows first; the audit engine treats every supplied partition label as a partition.

Always re-audit the selected partitions. The splitter guarantees disjointness only for configured scientific keys and declared groups; it cannot discover undeclared sample relationships. The API leaves the input unchanged and reports every row exactly once across train, test, and excluded lists. Neither train nor test can be empty.

The dataset fingerprint hashes an ordered, index-free pandas JSON representation with 15-digit float precision. It is a reproducibility aid, not a byte-for-byte identity guarantee for arbitrary DataFrames. The CSV command also hashes the exact input file bytes. Row order affects the fingerprint and seeded assignments. Reports record configuration and installed NumPy, pandas, and RDKit versions where available.

CSV inputs preserve all fields as strings, including leading-zero identifiers and literal `NA` values. Blank cells are considered missing. Numeric conversion is confined to configured numeric checks and extrapolation. Laboratory holdout values in JSON should therefore be strings for the CSV interface.

## Examples

`examples/electrolytes.csv` is entirely synthetic and deliberately contains duplicate records, partition overlap, inconsistent units, a negative absolute temperature, and missing provenance. Its SMILES are illustrative scaffold inputs, not claims about real electrolyte formulations. It is a software demonstration, not a scientific benchmark.

Each strategy has its own JSON configuration under `examples/`. `python examples/demo.py` runs the audit, all five strategies, and a re-audit showing that composition grouping removes the demonstrated partition overlaps while leaving unrelated data-quality findings visible.
