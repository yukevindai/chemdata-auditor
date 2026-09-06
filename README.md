# ChemData Auditor + SciSplit

Building tools to make scientific machine learning more trustworthy: audit the data, design meaningful splits, and understand what a model's performance actually tells you.

This repository brings together two complementary tools for chemistry, materials science, and chemical engineering:

- **ChemData Auditor** inspects scientific datasets for leakage, duplicated samples, unit errors, impossible values, sparse regions, and provenance gaps.
- **SciSplit** generates chemically meaningful train/test splits: composition holdout, scaffold holdout, laboratory holdout, time split, and extrapolation split.

> **Status:** Expanded ChemData Auditor implementation (v0.2.0), under review. SciSplit currently retains its initial five-strategy API. Includes a library, CSV command-line interface, JSON reports, synthetic examples, and tests. Findings support investigation; they do not certify scientific validity.

## Quick start

Requires Python 3.10 or newer. From the repository root:

```bash
python -m pip install -e '.[dev,chem]'
python -m pytest -q
python examples/demo.py
```

The optional `chem` extra installs RDKit for scaffold splitting. Use `python -m pip install -e .` for the audit engine and the other four strategies.

Audit the deliberately flawed synthetic example and generate a composition split:

```bash
python -m chemdata_auditor audit examples/electrolytes.csv --config examples/audit.json --output outputs/audit.json
python -m chemdata_auditor split examples/electrolytes.csv --config examples/composition.json --output outputs/composition.json
```

Installed command aliases are `chemdata audit`, `chemdata split`, and `scisplit`. Output files must be new paths. To fail a pipeline on audit errors, add `--fail-on error`; `--fail-on warning` also fails on warnings. Exit codes: `0` completed, `1` audit severity threshold reached (report still written), `2` invalid input/configuration or I/O failure.

See the [complete Auditor reference](docs/AUDITOR.md) for missingness, conflicts, unit conversion, constraints, molecular similarity, multivariate density, optional plugins, and HTML/Markdown reports.

See [configuration and scientific assumptions](docs/configuration.md) for every option and [review notes](docs/REVIEW.md) for implementation scope and known limits.

### Python API

```python
import pandas as pd
from chemdata_auditor import AuditConfig, SplitConfig, audit, split

df = pd.read_csv("examples/electrolytes.csv", dtype=str, keep_default_na=False)
report = audit(df, AuditConfig(
    bounds={"temperature": [0, None]},
    provenance_columns=["source", "lab"],
))
print(report.to_json())

result = split(df, SplitConfig(
    strategy="composition",
    columns=["ec_fraction", "dmc_fraction"],
    group_columns=["sample_id"],
    test_size=0.34,
    seed=42,
))
train = df.iloc[result.train].copy()
test = df.iloc[result.test].copy()
df["new_split"] = result.assignments()
```

Row references are **zero-based positions**, independent of the DataFrame index. Split reports include train, test, and explicitly excluded positions, actual partition sizes, overlap diagnostics, configuration, dependency versions, and a dataset fingerprint. Audit reports list checks run and skipped. CLI reports also record a SHA-256 digest of the original CSV bytes.

## Why this matters

Scientific ML results can look impressive because the dataset or evaluation split is flawed. Duplicated measurements, inconsistent units, and related samples appearing in both training and test sets can make reported performance misleading.

Random splits can reward memorization of familiar molecules, formulations, or experimental conditions. They do not necessarily answer the question that matters in practice: **will this model work on a new chemical system, in another laboratory, or on a future experiment?**

ChemData Auditor and SciSplit are intended to help researchers catch these problems before drawing conclusions from model scores.

## ChemData Auditor

**Purpose:** Inspect the dataset and flag issues that could undermine scientific conclusions.

### Implemented checks

| Check | What it should flag |
| --- | --- |
| Data leakage | Duplicate or declared group overlap across partitions; declared unavailable features, target-as-feature, and exact target copies. |
| Duplicated samples | Exact repeats, conflicting observation identities, numeric near duplicates, and canonical/similar molecular structures. |
| Unit errors | Missing/unknown units, dimensional incompatibility, and explicit conversions of a working copy for configured numeric checks. |
| Impossible values | Values that violate configured physical or experimental constraints, such as negative absolute temperatures or fractions outside their allowed range. |
| Sparse regions | Low-count or empty intervals and sparse multivariate neighborhoods with explicit distance scales. |
| Provenance gaps | Missing source references, sample identifiers, laboratory metadata, measurement methods, or transformation history. |

**Why it matters:** A strong model score cannot compensate for unreliable data. An audit should make potential problems visible and traceable so researchers can investigate them.

Findings identify affected row positions and columns, explain the check and its assumptions, and suggest a next step. Domain-dependent checks need user-supplied constraints and metadata; a flagged value is a reason to investigate, not automatic proof of an error.

### Expanded audit capabilities

The Auditor also reports missing values, constant columns, suspicious identifiers, target-proxy warnings, composition totals, conditional chemical constraints, chronological violations, and weak provenance. Every result preserves affected rows, evidence, severity, actions, and reproducible configuration. Optional presets cover batteries, molecules, materials, and process data. Use `--format json`, `--format markdown`, or `--format html`.

## SciSplit

**Purpose:** Generate train/test splits that match the scientific generalization question.

### Implemented split strategies

| Strategy | How it separates the data | What it tests |
| --- | --- | --- |
| Composition holdout | Hold out complete groups defined by exact composition columns or supplied formulation-family labels. | Performance on unseen compositions under the supplied grouping. |
| Scaffold holdout | Group molecules by a defined structural scaffold and hold out entire scaffold groups. | Transfer to unfamiliar molecular cores. |
| Laboratory holdout | Hold out all records from selected laboratories. | Transfer across experimental environments and practices. |
| Time split | Train on earlier observations and test on later observations using a declared timestamp and cutoff. | Performance on future observations. |
| Extrapolation split | Hold out values strictly above or below an explicit threshold in one numeric dimension. | Performance beyond the training range in that dimension. |

**Why it matters:** Random splits can test familiarity more than scientific generalization. The split strategy should reflect the intended use of the model, and its assumptions should be explicit.

Outputs include reproducible split assignments, configuration, group overlap counts, and observed train/test ranges for time and extrapolation splits. Declared sample groups stay together, including transitive relationships across multiple grouping columns. Boundary-crossing groups cause an error unless explicitly excluded in full. Preprocessing learned from data must be fitted on the training partition only; these tools do not fit preprocessing or models.

Scaffold grouping uses RDKit's [Bemis–Murcko scaffold implementation](https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html). It ignores chirality, groups all acyclic molecules together, and rejects invalid or disconnected SMILES. Standardization of salts and tautomers must be an explicit upstream decision.

## How the tools fit together

1. **Audit the dataset** with ChemData Auditor to identify quality, provenance, and grouping issues.
2. **Define the scientific question:** new compositions, molecular scaffolds, laboratories, future observations, or conditions beyond the training range.
3. **Generate the split** with SciSplit using the required metadata and grouping rules.
4. **Audit the resulting partitions** for duplicated samples, group overlap, and coverage gaps.
5. **Evaluate and report** model performance alongside the audit findings and split configuration.

For example, in an electrolyte dataset with multiple cycling measurements from each cell, a random row split could place measurements from the same cell in both training and test sets. Keeping each cell's measurements together addresses that overlap; holding out entire formulations asks the harder question of whether the model transfers to unseen electrolyte compositions.

## License

[MIT](LICENSE)
