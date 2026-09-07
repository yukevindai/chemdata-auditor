# ChemData Auditor + SciSplit

Building tools to make scientific machine learning more trustworthy: audit the data, design meaningful splits, and understand what a model's performance actually tells you.

This repository brings together two complementary tools for chemistry, materials science, and chemical engineering:

- **ChemData Auditor** inspects scientific datasets for leakage, duplicated samples, unit errors, impossible values, sparse regions, and provenance gaps.
- **SciSplit** generates train/validation/test partitions using formulation, composition, scaffold, publication, laboratory, temporal, cluster, and extrapolation holdouts, with random allocation as a comparison baseline.

> **Status:** Expanded toolkit implementation (v0.3.0), under review. ChemData Auditor and SciSplit cover the dataset-audit and scientific-evaluation workflows described below. Includes a library, CSV command-line interface, JSON reports, synthetic examples, and tests. Findings support investigation; they do not certify scientific validity.

## Quick start

Requires Python 3.10 or newer. From the repository root:

```bash
python -m pip install -e '.[dev,chem]'
python -m pytest -q
python examples/demo.py
```

The optional `chem` extra installs RDKit for scaffold splitting. Use `python -m pip install -e .` for the audit engine and the strategies that do not use molecular structures.

Audit the deliberately flawed synthetic example and generate a composition split:

```bash
python -m chemdata_auditor audit examples/electrolytes.csv --config examples/audit.json --output outputs/audit.json
python -m chemdata_auditor split examples/electrolytes.csv --config examples/composition.json --output outputs/composition.json
```

Installed command aliases are `chemdata audit`, `chemdata split`, and `scisplit`. Output files must be new paths. To fail a pipeline on audit errors, add `--fail-on error`; `--fail-on warning` also fails on warnings. Exit codes: `0` completed, `1` audit severity threshold reached (report still written), `2` invalid input/configuration or I/O failure.

See the [complete Auditor reference](docs/AUDITOR.md) for missingness, conflicts, unit conversion, constraints, molecular similarity, multivariate density, optional plugins, and HTML/Markdown reports.

See the [SciSplit reference](docs/SCISPLIT.md) for three-way partitions, strategy comparison, overlap, distribution shift, and training-domain diagnostics.

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
    test_size=0.2,
    validation_size=0.2,
    seed=42,
))
train = df.iloc[result.train].copy()
validation = df.iloc[result.validation].copy()
test = df.iloc[result.test].copy()
df["new_split"] = result.assignments()
```

Row references are **zero-based positions**, independent of the DataFrame index. Split reports include train, validation, test, and explicitly excluded positions, actual partition sizes, overlap diagnostics, configuration, dependency versions, and a dataset fingerprint. Audit reports list checks run and skipped. CLI reports also record a SHA-256 digest of the original CSV bytes.

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

**Purpose:** Generate and compare train/validation/test designs that match the scientific generalization question.

### Implemented split strategies

| Strategy | How it separates the data | What it tests |
| --- | --- | --- |
| Formulation holdout | Hold out declared formulation identities or families. | Transfer to unseen formulations. |
| Composition holdout | Hold out joint composition keys with decimal normalization and optional explicit rounding. | Performance on unseen compositions under the declared representation. |
| Scaffold holdout | Group molecules by a defined structural scaffold and hold out entire scaffold groups. | Transfer to unfamiliar molecular cores. |
| Publication holdout | Hold out complete source/publication groups. | Transfer across reported studies. |
| Laboratory holdout | Hold out all records from selected laboratories. | Transfer across experimental environments and practices. |
| Time split | Train on earlier observations and test on later observations using a declared timestamp and cutoff. | Performance on future observations. |
| Cluster holdout | Hold out K-means clusters in explicitly scaled covariate space. | Transfer across defined covariate clusters. |
| Extrapolation split | Hold out values beyond a threshold or a multivariate box. | Performance beyond declared numeric training/development regions. |
| Random baseline | Seeded row or independent-group allocation. | Comparison within the sampled population. |

**Why it matters:** Random splits can test familiarity more than scientific generalization. The split strategy should reflect the intended use of the model, and its assumptions should be explicit.

Outputs include reproducible split assignments, configuration, group overlap counts, and observed train/test ranges for time and extrapolation splits. Declared sample groups stay together, including transitive relationships across multiple grouping columns. Boundary-crossing groups cause an error unless explicitly excluded in full. Preprocessing learned from data must be fitted on the training partition only; these tools do not fit preprocessing or models.

Scaffold grouping uses RDKit's [Bemis–Murcko scaffold implementation](https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html). It ignores chirality, groups all acyclic molecules together, and rejects invalid or disconnected SMILES. Standardization of salts and tautomers must be an explicit upstream decision.

### Compare evaluation designs

```bash
python -m chemdata_auditor split examples/electrolytes.csv --config examples/split-three-way.json --output outputs/three-way.json --assignments outputs/assignments.csv
python -m chemdata_auditor compare examples/electrolytes.csv --config examples/compare.json --output outputs/comparison.html --format html
python -m chemdata_auditor diagnose examples/electrolytes.csv --config examples/diagnostics.json --split-report outputs/three-way.json --output outputs/diagnostics.md --format markdown
```

Every split explains its generalization scope and limitations. Diagnostics measure exact/group/near-duplicate overlap across all partition pairs, numeric and categorical distribution changes, and geometric distance from the training domain. Training-domain ranges, scales, and distance calibration use training rows only. Optional similarity grouping keeps related numeric or molecular samples together. Comparison reports preserve infeasible designs as explicit errors and provide no invented model-performance score.

## How the tools fit together

1. **Audit the dataset** with ChemData Auditor to identify quality, provenance, and grouping issues.
2. **Define the scientific question:** new compositions, molecular scaffolds, laboratories, future observations, or conditions beyond the training range.
3. **Generate the split** with SciSplit using the required metadata and grouping rules.
4. **Audit the resulting partitions** for duplicated samples, group overlap, and coverage gaps.
5. **Evaluate and report** model performance alongside the audit findings and split configuration.

For example, in an electrolyte dataset with multiple cycling measurements from each cell, a random row split could place measurements from the same cell in both training and test sets. Keeping each cell's measurements together addresses that overlap; holding out entire formulations asks the harder question of whether the model transfers to unseen electrolyte compositions.

## License

[MIT](LICENSE)
