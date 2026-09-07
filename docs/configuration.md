# Configuration and scientific assumptions

The complete configuration references are separated by tool:

- [ChemData Auditor](AUDITOR.md): missingness, identity/conflicts, leakage, dimensional units, physical and compositional constraints, conditional/chronological rules, similarity, coverage, provenance, plugins, and reports.
- [SciSplit](SCISPLIT.md): scientific train/validation/test allocation, grouping, boundary semantics, strategy comparison, distribution shifts, training-domain diagnostics, and exports.

Both tools operate locally and use explicit JSON-compatible configurations. Neither modifies source data nor trains a predictive model. All row references are zero-based positions in the original input; use `df.iloc`.

CSV parsing and hashing use the same immutable byte snapshot. CLI reports contain its SHA-256 digest. The Python API fingerprint hashes an ordered, index-free pandas JSON representation with 15-digit float precision; it is a reproducibility aid rather than a byte-for-byte identity guarantee for arbitrary DataFrames. Reports record dependency versions and effective configuration.

For a three-way split, specify `validation_size` for seeded designs or explicit validation values/boundaries for scientific boundary designs. The default validation fraction remains zero for compatibility with the original API. Fractions apply to connected groups, so actual row fractions may differ.

The examples are synthetic software demonstrations. Successful checks and tests do not certify a scientific claim. Report the dataset assumptions, split question, diagnostic findings, exclusions, and interpretation limits alongside model metrics.
