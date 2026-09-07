# SciSplit: scientific evaluation designs and diagnostics

SciSplit generates train, validation, and test assignments, preserves declared sample relationships, and describes what kind of generalization each design can test. It compares designs through transparent diagnostics. It does not train models or manufacture a model-performance score.

## Run the complete workflow

```bash
python -m pip install -e '.[dev,chem]'
python -m chemdata_auditor split examples/electrolytes.csv --config examples/split-three-way.json --output outputs/three-way.json --assignments outputs/assignments.csv
python -m chemdata_auditor compare examples/electrolytes.csv --config examples/compare.json --output outputs/comparison.html --format html
python -m chemdata_auditor diagnose examples/electrolytes.csv --config examples/diagnostics.json --split-report outputs/three-way.json --output outputs/diagnostics.md --format markdown
```

Outputs must use new paths. Assignment CSVs contain `row_position` and `partition`; use those positions with the original input, whose byte hash is recorded in the JSON report. The `diagnose` command rejects reports from a different dataset snapshot. An infeasible design in `compare` is retained as an explicit error while other designs continue; the CLI returns exit code 1 if any design fails. Invalid top-level inputs return 2.

The comparison example covers random rows, formulation, composition, scaffold, publication, laboratory, temporal, cluster, and extrapolation designs. The data, publication labels, and molecule/formulation associations are synthetic illustrations. They are not experimental evidence.

## Python API

```python
from chemdata_auditor import (
    SplitConfig, DiagnosticsConfig, split, diagnose, compare_splits,
)

config = SplitConfig(
    strategy="composition",
    columns=["ec_fraction", "dmc_fraction"],
    group_columns=["sample_id"],
    validation_size=0.2,
    test_size=0.2,
    seed=42,
    diagnostics={
        "numeric_columns": ["capacity"],
        "categorical_columns": ["lab"],
        "domain_columns": ["ec_fraction", "dmc_fraction"],
    },
)
result = split(df, config)
train = df.iloc[result.train].copy()
validation = df.iloc[result.validation].copy()
test = df.iloc[result.test].copy()
labels = result.assignments()  # also includes 'excluded' if explicitly requested

comparison = compare_splits(
    df,
    {"composition": config, "random": SplitConfig("random", validation_size=0.2)},
    DiagnosticsConfig(group_columns=["sample_id"], numeric_columns=["capacity"]),
)
```

Every row belongs to exactly one partition. Train and test must be nonempty, as must validation when requested. The default `validation_size=0` preserves the original two-way API; specify a fraction, explicit validation labels, or validation boundaries for a three-way design. The supplied DataFrame is not modified. Missing scientific keys and grouping metadata are errors, not inferred identities.

## What each strategy tests

| Strategy | Allocation | Interpretation limits |
| --- | --- | --- |
| `random` | Seeded allocation of rows or declared independent connected groups. `columns` must be empty. | Reference estimate within the sampled population; does not establish new-chemistry transfer. |
| `formulation` | Hold out complete identities/families in one supplied formulation column. | Labels may hide chemically equivalent or near-identical formulations. |
| `composition` | Joint keys over one or more composition columns, with numeric representation normalization and optional rounding. | Disjoint compositions need not be chemically distant or outside the training domain. |
| `scaffold` | RDKit Bemis–Murcko groups from one SMILES column. | Ignores scaffold chirality; all acyclic compounds share a group. Does not ensure fingerprint dissimilarity. |
| `publication` | Hold out entire publication/source labels. | Publications may share experiments or reused datasets; audit provenance and sample overlap. |
| `laboratory` | Hold out entire laboratory labels. | Laboratory effects can be confounded with chemistry, instrumentation, and procedures. |
| `time` / `temporal` | Chronologically ordered partitions with explicit cutoffs. | Order alone does not remove repeated-sample overlap or establish immunity to future drift. |
| `cluster` | K-means clusters of declared numeric covariates, followed by whole-cluster allocation. | Covariates from all rows design the benchmark; cluster geometry depends on representation and scales. |
| `extrapolation` | Explicit high/low thresholds or an axis-aligned multivariate region. | Tests the selected dimensions and boundary, not arbitrary chemical extrapolation. |

Every successful report contains `generalization.supports`, `does_not_establish`, and `conditional_on`, plus a structured scope finding. Scientific claims remain conditional on valid data, provenance, independent samples, and unavailable-at-prediction-time information being excluded.

## Split configuration

| Field | Default and meaning |
| --- | --- |
| `strategy` | Required; one of the strategies above. |
| `columns` | `[]`; required for scientific strategies, empty for random. Composition, cluster, and box extrapolation can use multiple columns. Other strategies use one. |
| `group_columns` | `[]`; each column independently constrains related rows to stay together. |
| `test_size`, `validation_size` | `0.2`, `0.0`; fractions of independent connected groups, not guaranteed row fractions. Fractions must leave training mass. |
| `seed` | `0`; integer in [0, 2^32). Used for seeded allocation and clustering. |
| `holdout_values`, `validation_values` | Optional disjoint explicit test/validation labels for formulation, publication, or laboratory. Validation labels require explicit test labels. Unknown labels fail. |
| `cutoff`, `validation_cutoff` | Explicit temporal test boundary and optional earlier train/validation boundary. Absolute ISO dates/timestamps only; naive timestamps are UTC. |
| `threshold`, `validation_threshold`, `direction` | Single-column extrapolation test threshold, optional inner validation threshold, and `high` or `low`. |
| `extrapolation_bounds`, `validation_bounds` | Alternative multivariate `[lower, upper]` bounds per column. The outer box contains train+validation; an optional inner box defines training. |
| `crossing_policy` | `error`; temporal/extrapolation only. `exclude` removes entire connected groups that cross any boundary. |
| `composition_decimals` | Unset; optional integer 0–15 for explicitly rounding numeric composition keys. |
| `n_clusters`, `cluster_scales` | `5` clusters; a positive explicit scale for every clustering column is required. |
| `group_near_duplicates` | `{}`; absolute numeric tolerances used to connect similar rows before allocation. |
| `group_smiles_column`, `group_molecular_similarity` | Optional canonical molecule grouping and optional Morgan Tanimoto similarity grouping. Invalid/disconnected SMILES fail. |
| `target_column`, `allow_target_in_split` | Optional outcome declaration; using it as a scientific key, group, or similarity feature is rejected unless explicitly allowed. Outcome-conditioned selection is disclosed. |
| `max_pair_comparisons` | 2,000,000 for grouping similarity calculations. Exceeding the budget raises an error. |
| `diagnostics` | A `DiagnosticsConfig` object represented as a JSON dictionary, described below. |

Explicit label and boundary strategies do not use fraction/seed allocation; nondefault `test_size`, `validation_size`, or `seed` settings are rejected for those designs. Configure validation with labels/cutoffs/bounds instead. Strategy-specific options used with another strategy are rejected.

### Grouping semantics

Rows are connected if they share a scientific key, any declared grouping identity, canonical molecular identity, or a configured numeric/molecular similarity edge. Connections are transitive. If A resembles B and B resembles C, all three stay together even if A and C are distant. Large components can make a requested design impossible. The tool reports this instead of breaking groups to obtain a desired fraction.

Seeded allocation shuffles the connected components, selects `ceil(test_size * component_count)` for test and `ceil(validation_size * component_count)` for validation, capped to preserve at least one component for each requested partition. Actual row fractions are reported. There must be at least two independent components, or three for three-way allocation. This is not stratification by target value.

Use globally meaningful group IDs. If laboratories reuse cell IDs, construct a lab-plus-cell composite identifier; otherwise identical local IDs will intentionally connect those rows. Declare numeric/molecular similarity grouping only when those relationships genuinely must stay together. Merely sharing a molecule does not make different experiments duplicate measurements.

Composition keys normalize decimal values, so `0.2` and `0.20` group together. Optional rounding uses decimal rounding, not a pairwise tolerance; nearby values on opposite bin boundaries can remain separate. Use `group_near_duplicates` when a tolerance relationship is required. Non-numeric composition/family labels remain exact strings. Units and chemical equivalence are never inferred; audit and normalize units first.

K-means uses explicit scales and `n_init=10`, `max_iter=300`, and the configured seed. All supplied covariates are used to define the benchmark clusters. That design fit is disclosed and must not be reused as model preprocessing. Clustering centers, labels, inertia, software versions, and configuration are saved. Target labels must not be clustering inputs unless the study explicitly defines an outcome-conditioned benchmark.

### Boundary examples

Time: train <= `validation_cutoff`; validation is later than that cutoff and <= `cutoff`; test is later than `cutoff`. Without a validation cutoff, train includes everything <= the test cutoff.

High extrapolation: train <= `validation_threshold`; validation lies above that boundary and <= `threshold`; test is strictly above `threshold`. Low extrapolation reverses the inequalities. Equality belongs to the inner partition. Validation boundaries must be strictly inside the development region.

Multivariate extrapolation: test is outside the outer `extrapolation_bounds` in any selected dimension; validation lies within that box but outside the optional inner `validation_bounds`; training lies inside the inner box (or the outer box when no validation is requested). Boundaries are inclusive inside. The inner box must be contained in the outer one.

Any connected sample group crossing a train/validation/test boundary causes an error, or is excluded in full when explicitly configured. Exclusion can bias the remaining population. Empty resulting partitions still cause an error.

## Diagnostic configuration

`diagnose(df, result, DiagnosticsConfig(...))` returns metrics and structured findings. `split` runs diagnostics automatically using its `diagnostics` dictionary, adding its declared grouping columns. Diagnostic columns can therefore also inspect relationships not enforced during allocation, which is useful when comparing random splits against scientific holdouts.

| Field | Meaning |
| --- | --- |
| `group_columns` | Group overlap counts and affected rows for every partition pair. Missing IDs produce not-evaluable findings. |
| `duplicate_columns` | Exact record comparison columns; default is all supplied columns. Choose observation keys to avoid irrelevant metadata concealing duplicates. |
| `near_duplicates` | Per-column absolute tolerances; count cross-partition numeric pairs and provide evidence. |
| `smiles_column`, `molecular_similarity` | Optional Morgan radius-2, 2048-bit, chirality-aware Tanimoto comparison. Invalid structures are reported. |
| `numeric_columns` | Empirical Kolmogorov–Smirnov distance, means, training-standardized mean difference, and invalid counts for validation/test versus train. |
| `categorical_columns` | Total variation in category frequencies, unseen categories, and their row fraction versus train. Missingness is a separate category. |
| `shift_threshold` | `0.2`; descriptive KS/total-variation threshold in (0,1]. It is not a p-value or significance test. |
| `domain_columns` | Numeric dimensions used to inspect training-domain coverage. |
| `domain_scales` | Optional explicit positive scale per domain column. Otherwise scales use training-only standard deviations, with scale 1 for a constant training dimension. |
| `domain_quantile` | `0.95`; quantile of training leave-one-out nearest-neighbor distances used to calibrate distance screening. |
| `domain_distance_threshold` | Optional nonnegative explicit distance cutoff instead of training calibration. |
| `max_pair_comparisons` | 2,000,000 per pairwise calculation. Grouping and diagnostic budgets are configured separately. |

Exact duplicate pair counts account for repeated occurrences on both sides. Similarity evidence includes at most 100 example pairs; counts and affected rows are complete. No pairs involving excluded rows contribute to overlap counts. No silent subsampling is performed.

Numeric shifts exclude invalid measurements and report those exclusions. KS distance is the largest absolute difference between the empirical cumulative distributions. Standardized mean difference divides the evaluation-minus-training mean by training standard deviation; it is undefined when training is constant, which is recorded explicitly. Categorical total variation is half the sum of absolute frequency differences over the union of categories.

Training-domain diagnostics fit ranges and scales on complete training rows only. Each finite validation/test sample is compared with its nearest training neighbor. A sample is flagged if it exceeds a training range in any dimension or its nearest-neighbor distance exceeds the configured/calibrated threshold. The report includes nearest training row, distance, range status, cutoff status, fitted scales/ranges, and all affected rows.

The default distance cutoff is a quantile of training leave-one-out distances; no evaluation data is used in that calibration. Repeated training samples can produce a zero cutoff. A single valid training row cannot calibrate neighbor distances, so that part is marked not evaluable unless an explicit cutoff is provided. Missing rows are reported, not imputed. This is a geometric applicability screen, not a prediction interval or a guarantee of chemical validity. Choose domain covariates that are available at prediction time.

## Comparing strategies

`compare_splits` accepts named `SplitConfig` objects or dictionaries and an optional shared `DiagnosticsConfig`. Shared diagnostic settings override per-strategy settings to keep comparisons consistent; allocation group constraints are still included. Each design records status, sizes, exclusions, finding counts, domain coverage, generalization scope, full assignments, and full evidence. Infeasible designs retain their error and configuration. There is no hidden fallback to a random split.

A low-shift split can be easy but scientifically irrelevant; a difficult held-out laboratory can be the appropriate deployment test. Use the reported scope and diagnostics to select the question, then train preprocessing/models on training data, tune using validation, and reserve test data for final evaluation. Do not repeatedly select a split because it yields a better test score.

Tests cover all strategies, three-way disjointness, numerical composition normalization, K-means reproducibility, multivariate boundaries, transitive similarity grouping, validation leakage, known distribution shifts, training-only domain calibration, invalid snapshots, failed comparison designs, and CLI exports.

Implementation references: [RDKit scaffolds](https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html), [RDKit fingerprints](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html), and [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).
