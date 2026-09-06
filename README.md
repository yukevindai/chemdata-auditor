# ChemData Auditor + SciSplit

Building tools to make scientific machine learning more trustworthy: audit the data, design meaningful splits, and understand what a model's performance actually tells you.

This repository brings together two complementary tools for chemistry, materials science, and chemical engineering:

- **ChemData Auditor** inspects scientific datasets for leakage, duplicated samples, unit errors, impossible values, sparse regions, and provenance gaps.
- **SciSplit** generates chemically meaningful train/test splits: composition holdout, scaffold holdout, laboratory holdout, time split, and extrapolation split.

> **Status:** Project scope and planned capabilities. This repository currently contains documentation and a license; the tools are not implemented yet.

## Why this matters

Scientific ML results can look impressive because the dataset or evaluation split is flawed. Duplicated measurements, inconsistent units, and related samples appearing in both training and test sets can make reported performance misleading.

Random splits can reward memorization of familiar molecules, formulations, or experimental conditions. They do not necessarily answer the question that matters in practice: **will this model work on a new chemical system, in another laboratory, or on a future experiment?**

ChemData Auditor and SciSplit are intended to help researchers catch these problems before drawing conclusions from model scores.

## ChemData Auditor

**Purpose:** Inspect the dataset and flag issues that could undermine scientific conclusions.

### Planned checks

| Check | What it should flag |
| --- | --- |
| Data leakage | Overlapping sample identities or related experimental groups across splits, and features that may expose information unavailable at prediction time. |
| Duplicated samples | Exact duplicates and likely repeated samples, including repeated records across data sources. |
| Unit errors | Missing, inconsistent, or incompatible units and suspicious scale differences. |
| Impossible values | Values that violate configured physical or experimental constraints, such as negative absolute temperatures or fractions outside their allowed range. |
| Sparse regions | Poorly represented regions of composition, property, or experimental-condition space. |
| Provenance gaps | Missing source references, sample identifiers, laboratory metadata, measurement methods, or transformation history. |

**Why it matters:** A strong model score cannot compensate for unreliable data. An audit should make potential problems visible and traceable so researchers can investigate them.

Planned findings should identify the affected records or columns, explain the check and its assumptions, and suggest a next step. Domain-dependent checks will need user-supplied constraints and metadata; a flagged value is a reason to investigate, not automatic proof of an error.

## SciSplit

**Purpose:** Generate train/test splits that match the scientific generalization question.

### Planned split strategies

| Strategy | How it separates the data | What it tests |
| --- | --- | --- |
| Composition holdout | Reserve selected compositions or formulation families for testing, keeping related samples together. | Performance on unseen compositions. |
| Scaffold holdout | Group molecules by a defined structural scaffold and hold out entire scaffold groups. | Transfer to unfamiliar molecular cores. |
| Laboratory holdout | Hold out all records from selected laboratories. | Transfer across experimental environments and practices. |
| Time split | Train on earlier observations and test on later observations using a declared timestamp and cutoff. | Performance on future observations. |
| Extrapolation split | Reserve a defined region beyond the training range in selected composition, property, or condition dimensions. | Performance outside the observed training domain. |

**Why it matters:** Random splits can test familiarity more than scientific generalization. The split strategy should reflect the intended use of the model, and its assumptions should be explicit.

Planned outputs include reproducible split assignments, the configuration used to generate them, and diagnostics for group overlap and data coverage. Related samples should stay together where the evaluation requires it. Preprocessing learned from data must be fitted on the training partition only.

## How the tools fit together

1. **Audit the dataset** with ChemData Auditor to identify quality, provenance, and grouping issues.
2. **Define the scientific question:** new compositions, molecular scaffolds, laboratories, future observations, or conditions beyond the training range.
3. **Generate the split** with SciSplit using the required metadata and grouping rules.
4. **Audit the resulting partitions** for duplicated samples, group overlap, and coverage gaps.
5. **Evaluate and report** model performance alongside the audit findings and split configuration.

For example, in an electrolyte dataset with multiple cycling measurements from each cell, a random row split could place measurements from the same cell in both training and test sets. Keeping each cell's measurements together addresses that overlap; holding out entire formulations asks the harder question of whether the model transfers to unseen electrolyte compositions.

## License

[MIT](LICENSE)
