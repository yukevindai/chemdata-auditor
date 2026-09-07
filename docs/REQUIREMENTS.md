# Expanded toolkit implementation coverage

This maps the requested product scope to executable code and validation. It does not claim that configurable checks detect every possible scientific flaw.

| Requirement | Implementation | Verification |
| --- | --- | --- |
| Missing, duplicate, conflicting, and suspicious records | `audit.py`, `audit_extensions.py` | Missingness, observation identity conflicts, repeated records, constant columns, identifier screening tests |
| Target leakage | Explicit unavailable inputs, target-as-feature, exact target copies, and high-correlation proxy warnings | Original audit tests and expanded proxy tests |
| Units and impossible values | Pint dimensions and internal conversions before configured numeric checks | Offset-temperature, concentration conversion, incompatible/unknown unit tests |
| Compositional and domain rules | Nonnegative sums, safe predicates, conditional and chronological relationships in `rules.py` | Missing evidence, composition errors, three-valued logic, chronology tests |
| Sparse regions and provenance | Marginal bins, scaled multivariate neighborhoods, required metadata and syntax/placeholder checks | Sparse neighborhoods, bins, weak/missing source tests |
| Numeric/molecular near duplicates | Bounded pair checks and canonical identities in `similarity.py` | Numeric and RDKit pair/identity tests with invalid-row evidence |
| Meaningful three-way partitions | Formulation, composition, scaffold, publication, laboratory, temporal, cluster, and extrapolation engines | Every strategy tested; partitions cover each row once and preserve declared constraints |
| Group and near-duplicate overlap | Pairwise train/validation/test diagnostics; optional transitive similarity grouping | Validation leakage and numeric/molecular grouping tests |
| Distribution shifts | Numeric empirical KS and training-standardized means; categorical total variation and unseen categories | Tests with known KS/TV distances |
| Outside-training-domain samples | Training-only ranges, scales, and leave-one-out neighbor-distance calibration | Constant training domain, joint coverage holes, and held-out-extreme invariance tests |
| Compare strategies and explain generalization | Named design reports, explicit infeasibility, scope/limitations per strategy | Comparison error handling, scope, summary, and rendering tests |
| Transparent reproducible evidence | Full row references, severity, actions, capped raw/pair examples, complete counts, hashes and effective config | JSON serialization, snapshot identity, CSV concurrent-edit, escaping tests |
| Local Python and CLI workflows | Audit, split, compare, diagnose, plugins, preset; JSON/Markdown/HTML and assignment CSV exports | End-to-end CLI tests and executable examples |
| Optional domain plugins | Battery, molecular, materials, process presets and explicit external entry points | Executable tests for all four built-in domains |

The Auditor expansion was submitted first as PR #2 and merged by the repository owner. PR #3 addresses its review edge cases. SciSplit follows as a separate PR on top of those fixes, sharing reporting and similarity infrastructure. This implementation workflow does not merge pull requests.

Limitations are intrinsic to the declared methods: source syntax cannot authenticate provenance; similarity cannot establish experimental equivalence; target correlation is not causal evidence; clustering on all covariates is benchmark design rather than model preprocessing; geometric domain distance is not predictive uncertainty. Dependency minimums have not been validated across every supported Python/platform combination.
