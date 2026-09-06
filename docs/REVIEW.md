> Historical review of v0.1.0. The expanded Auditor implementation is documented in [AUDITOR.md](AUDITOR.md); the limitations below describe the original foundation.

# Initial implementation review

This implementation is proposed for review on `feat/audit-and-split-foundation`, based on commit `7c7df7e5a897c46358d48cb9197319c8409810b7`. The branch is being submitted through a pull request at Kevin's request. Merging remains pending review.

## What is ready

- An installable Python package exposing `audit()` and `split()`.
- ChemData Auditor checks for exact duplicates, cross-partition duplicate/group overlap, explicit feature leakage, declared unit-label mismatches, configured physical bounds, sparse one-dimensional intervals, and missing provenance metadata.
- SciSplit implements composition, scaffold, laboratory, time, and one-dimensional extrapolation holdouts.
- Declared related samples remain together, including transitive relationships across multiple grouping columns.
- Time/extrapolation boundary conflicts raise an error by default; explicit exclusion removes entire conflicting groups.
- CSV command-line entry points, JSON configurations and reports, positional row assignments, reproducibility metadata, and non-overwriting report outputs.
- A deliberately flawed synthetic example with all five split configurations and an audit/re-audit demonstration.
- Documentation of configuration, assumptions, edge cases, and interpretation limits.

## Start reviewing here

| File | What to review |
| --- | --- |
| `src/chemdata_auditor/audit.py` | Audit rules, severity, affected rows, and suggestions. |
| `src/chemdata_auditor/split.py` | Scientific grouping, boundary rules, and exclusion policy. |
| `src/chemdata_auditor/common.py` | Input validation, row identity, and reproducibility metadata. |
| `src/chemdata_auditor/cli.py` | CSV parsing, identifier preservation, exit codes, and output behavior. |
| `examples/` | A runnable demonstration and editable configurations. |
| `tests/` | Tests of leakage, scientific boundaries, and command behavior. |
| `docs/configuration.md` | Complete configuration reference and scientific limitations. |

## Validation completed

Tested with Python 3.12, NumPy 2.3.5, pandas 2.2.3, and RDKit 2026.3.6.

- Editable package installation with `dev` and `chem` extras succeeded.
- `python -m pytest -q`: **52 passed**, including actual RDKit scaffold tests and the missing-dependency error path.
- `python examples/demo.py`: all five strategies completed; composition re-audit found no duplicate or declared sample-group overlap.

Synthetic example results:

| Strategy | Train rows | Test rows | Excluded rows |
| --- | ---: | ---: | ---: |
| Composition | 6 | 6 | 0 |
| Laboratory | 8 | 4 | 0 |
| Scaffold | 10 | 2 | 0 |
| Time | 6 | 6 | 0 |
| Extrapolation | 8 | 4 | 0 |

The initial audit detects repeated records, duplicate/group overlap, missing/mismatched units, an out-of-bounds measurement, missing provenance, and sparse bins. Fixing the split leaves independent data quality findings visible.

## Decisions to review

1. **Configuration before inference.** Domain-dependent rules use declared metadata and bounds. No unit conversion, inferred group identity, or automatic scientific correction is performed.
2. **Group-based allocation.** For seeded splits, `test_size` is a fraction of connected groups. Unequal group sizes can produce a very different row fraction; reports show the actual size.
3. **Exact composition grouping.** Supply normalized numeric representations or validated family labels. CSV values remain strings, so `0.2` and `0.20` differ unless normalized upstream.
4. **Strict scaffold policy.** RDKit Bemis–Murcko scaffolds ignore chirality; all acyclic molecules share one group. Invalid/disconnected SMILES are rejected, and tautomer normalization is outside this version.
5. **Strict boundaries.** Time/extrapolation groups crossing the boundary are rejected or explicitly excluded in full. The algorithm does not relax the boundary to force a requested test fraction.

## Limits and next work

This is a functional foundation, not a validated scientific benchmark or a comprehensive dataset certification system. Near-duplicate chemistry matching, dimensional unit validation/conversion, multivariate density and extrapolation, automatic provenance verification, task-specific scientific presets, and interactive reports remain future work.

The implementation has been tested locally on synthetic cases. A real, provenance-documented dataset and expert review of domain constraints are needed before making empirical claims about its usefulness. The declared Python/dependency minimums have not yet been tested as a compatibility matrix.

To try it from the repository root:

```bash
python -m pip install -e '.[dev,chem]'
python -m pytest -q
python examples/demo.py
python -m chemdata_auditor audit examples/electrolytes.csv --config examples/audit.json --output outputs/my-audit.json
```

To inspect the implementation, use the pull request's Files changed tab or run `git diff origin/main...HEAD` from the feature branch. Merging remains a separate step after review.
