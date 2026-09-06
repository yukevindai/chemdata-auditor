"""Scientific holdouts with explicit grouping and boundary semantics."""

import math
import random
import re
from numbers import Number
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .common import JsonResult, finite_number, frame, metadata, names, numeric, require


@dataclass
class SplitConfig:
    strategy: str
    columns: list[str]
    group_columns: list[str] = field(default_factory=list)
    test_size: float = 0.2
    seed: int = 0
    holdout_values: list | None = None
    cutoff: str | None = None
    threshold: float | None = None
    direction: str = "high"
    crossing_policy: str = "error"

    def __post_init__(self):
        if self.strategy not in {"composition", "scaffold", "laboratory", "time", "extrapolation"}:
            raise ValueError("Unknown split strategy.")
        names(self.columns, "columns")
        names(self.group_columns, "group_columns")
        if not self.columns or (self.strategy != "composition" and len(self.columns) != 1):
            raise ValueError("Supply one split column, or one or more composition columns.")
        finite_number(self.test_size, "test_size")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be strictly between 0 and 1.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
        if self.direction not in {"high", "low"} or self.crossing_policy not in {"error", "exclude"}:
            raise ValueError("direction must be high/low and crossing_policy error/exclude.")
        if self.strategy == "time" and (not isinstance(self.cutoff, str) or not self.cutoff.strip()):
            raise ValueError("A time split requires an explicit ISO timestamp cutoff.")
        if self.strategy == "time" and not re.match(r"^\d{4}-\d{2}-\d{2}(?:T| |$)", self.cutoff):
            raise ValueError("cutoff must be an absolute ISO date or timestamp, not a relative date.")
        if self.strategy != "time" and self.cutoff is not None:
            raise ValueError("cutoff is only valid for time splits.")
        if self.strategy == "extrapolation":
            finite_number(self.threshold, "threshold")
        elif self.threshold is not None or self.direction != "high":
            raise ValueError("threshold and direction are only valid for extrapolation splits.")
        if self.holdout_values is not None:
            if self.strategy != "laboratory" or not isinstance(self.holdout_values, list) or not self.holdout_values:
                raise ValueError("holdout_values must be a nonempty list for laboratory holdout.")
            if any(not isinstance(v, (str, int, float)) or isinstance(v, bool) for v in self.holdout_values):
                raise ValueError("Laboratory holdout values must be scalar strings or numbers.")
            for value in self.holdout_values:
                if isinstance(value, (int, float)):
                    finite_number(value, "holdout value")
            if len(set(self.holdout_values)) != len(self.holdout_values):
                raise ValueError("holdout_values contains duplicates.")
        if self.strategy in {"time", "extrapolation"} or self.holdout_values is not None:
            if self.test_size != 0.2 or self.seed != 0:
                raise ValueError("test_size and seed do not apply to explicit boundary or laboratory holdouts.")
        if self.strategy not in {"time", "extrapolation"} and self.crossing_policy != "error":
            raise ValueError("crossing_policy only applies to time and extrapolation boundaries.")


@dataclass
class SplitResult(JsonResult):
    train: list[int]
    test: list[int]
    excluded: list[int]
    diagnostics: dict
    metadata: dict

    def assignments(self):
        """Return partition labels in original row order."""
        labels = ["excluded"] * (len(self.train) + len(self.test) + len(self.excluded))
        for label, rows in (("train", self.train), ("test", self.test)):
            for row in rows:
                labels[row] = label
        return labels


def _components(n, key_sets):
    # Union-find joins rows sharing ANY declared group, including transitive links.
    parent = list(range(n))

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for keys in key_sets:
        first = {}
        for i, key in enumerate(keys):
            if key in first:
                parent[root(i)] = root(first[key])
            else:
                first[key] = i
    buckets = {}
    for i in range(n):
        buckets.setdefault(root(i), []).append(i)
    return list(buckets.values())


def _scaffolds(series):
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    except ImportError as exc:
        raise ValueError('Scaffold splitting needs RDKit: pip install "chemdata-auditor[chem]"') from exc
    keys, cache = [], {}
    for row, smiles in enumerate(series):
        if not isinstance(smiles, str):
            raise ValueError(f"SMILES at row {row} must be a string.")
        if smiles not in cache:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                raise ValueError(f"Invalid SMILES at row {row}: {smiles!r}")
            if len(Chem.GetMolFrags(mol)) != 1:
                raise ValueError(f"Disconnected SMILES at row {row}; explicitly standardize salts/mixtures first.")
            cache[smiles] = MurckoScaffoldSmiles(mol=mol, includeChirality=False) or "<acyclic>"
        keys.append(cache[smiles])
    return keys


def split(data, config):
    """Return row positions; no preprocessing is learned and no rows are modified."""
    cfg = config
    data = frame(data)
    require(data, cfg.columns + cfg.group_columns, complete=True)
    n = len(data)
    group_keys = [data[c].tolist() for c in cfg.group_columns]
    notes = []
    if not cfg.group_columns:
        notes.append("No sample-group columns supplied; repeated experiments across different split keys cannot be identified.")
    keys, cut_values = None, None
    diagnostics = {"strategy": cfg.strategy, "warnings": notes}

    if cfg.strategy in {"composition", "laboratory", "scaffold"}:
        if cfg.strategy == "scaffold":
            keys = _scaffolds(data[cfg.columns[0]])
            notes.append("Bemis–Murcko scaffolds ignore chirality; all acyclic molecules share one group. No tautomer normalization is performed.")
        elif cfg.strategy == "composition":
            keys = list(data[cfg.columns].itertuples(index=False, name=None))
            notes.append("Composition keys use exact supplied values. Normalize units and define formulation families before splitting.")
        else:
            keys = data[cfg.columns[0]].tolist()
        components = _components(n, [keys, *group_keys])
        if cfg.holdout_values is not None:
            unknown = set(cfg.holdout_values) - set(keys)
            if unknown:
                raise ValueError(f"Unknown laboratory holdout values: {sorted(map(str, unknown))}")
            labels = ["test" if key in cfg.holdout_values else "train" for key in keys]
            if any(len({labels[i] for i in group}) > 1 for group in components):
                raise ValueError("Declared sample groups connect held-out laboratories to training laboratories.")
            diagnostics["selection"] = "explicit laboratory values"
        else:
            if len(components) < 2:
                raise ValueError("At least two independent groups are required for a nonempty holdout.")
            shuffled = list(components)
            random.Random(cfg.seed).shuffle(shuffled)
            count = min(len(shuffled) - 1, max(1, math.ceil(len(shuffled) * cfg.test_size)))
            held = {i for group in shuffled[:count] for i in group}
            labels = ["test" if i in held else "train" for i in range(n)]
            diagnostics.update(selection="seeded connected-group holdout", requested_test_group_fraction=cfg.test_size,
                               heldout_components=count)
        diagnostics.update(independent_components=len(components), scientific_groups=len(set(keys)))
    else:
        col = cfg.columns[0]
        if cfg.strategy == "time":
            # Naive timestamps are interpreted as UTC; timezone-aware values are normalized.
            if any(isinstance(value, Number) for value in data[col]):
                raise ValueError("Numeric timestamps are ambiguous; supply ISO timestamps.")
            if any(isinstance(value, str) and not re.match(r"^\d{4}-\d{2}-\d{2}(?:T| |$)", value) for value in data[col]):
                raise ValueError("Time values must use absolute ISO dates or timestamps.")
            cut_values = pd.to_datetime(data[col], errors="coerce", utc=True, format="mixed")
            cutoff = pd.to_datetime(cfg.cutoff, errors="coerce", utc=True)
            if pd.isna(cutoff) or cut_values.isna().any():
                raise ValueError("Time split requires valid timestamps and a valid cutoff.")
            held = (cut_values > cutoff).to_numpy()
            diagnostics["boundary"] = {"cutoff_utc": cutoff.isoformat(), "train": "<= cutoff", "test": "> cutoff"}
            notes.append("Naive timestamps are interpreted as UTC. The timestamp must reflect availability at prediction time.")
        else:
            cut_values = numeric(data[col])
            if not np.isfinite(cut_values).all():
                raise ValueError("Extrapolation values must be finite numeric measurements.")
            held = cut_values > cfg.threshold if cfg.direction == "high" else cut_values < cfg.threshold
            diagnostics["boundary"] = {"threshold": cfg.threshold, "direction": cfg.direction, "equality": "train"}
            notes.append("This tests extrapolation in one configured dimension, not distance from a multivariate training domain.")
        labels = ["test" if x else "train" for x in held]
        components = _components(n, group_keys)
        crossing = [group for group in components if len({labels[i] for i in group}) > 1]
        if crossing and cfg.crossing_policy == "error":
            rows = [i for group in crossing for i in group]
            raise ValueError(f"Sample groups cross the split boundary (rows {rows[:20]}). Use crossing_policy='exclude' to exclude entire crossing groups.")
        for group in crossing:
            for i in group:
                labels[i] = "excluded"
        diagnostics["excluded_crossing_groups"] = len(crossing)
        if crossing:
            notes.append("Entire boundary-crossing groups were excluded; inspect selection bias before interpreting scores.")

    train, test, excluded = ([i for i, label in enumerate(labels) if label == target] for target in ("train", "test", "excluded"))
    if not train or not test:
        raise ValueError("The requested split leaves an empty training or test partition.")
    overlaps = {}
    for col in cfg.group_columns:
        overlaps[col] = len(set(data.iloc[train][col]) & set(data.iloc[test][col]))
    diagnostics["group_overlap_counts"] = overlaps
    if keys is not None:
        diagnostics["scientific_group_overlap"] = len({keys[i] for i in train} & {keys[i] for i in test})
    if any(overlaps.values()) or diagnostics.get("scientific_group_overlap", 0):
        raise RuntimeError("Internal invariant failed: groups overlap across partitions.")
    if cut_values is not None:
        ranges = {}
        for label, rows in (("train", train), ("test", test)):
            selected = cut_values.iloc[rows] if cfg.strategy == "time" else cut_values[rows]
            ranges[label] = ([selected.min().isoformat(), selected.max().isoformat()] if cfg.strategy == "time"
                             else [float(selected.min()), float(selected.max())])
        diagnostics["observed_ranges"] = ranges
    diagnostics.update(n_train=len(train), n_test=len(test), n_excluded=len(excluded),
                       actual_test_row_fraction=len(test) / (len(train) + len(test)))
    return SplitResult(train, test, excluded, diagnostics, metadata(data, cfg))
