"""Scientific holdouts with explicit grouping and boundary semantics."""

from dataclasses import dataclass, field

from .common import JsonResult
from .split_config import SplitConfig


@dataclass
class SplitResult(JsonResult):
    train: list[int]
    test: list[int]
    excluded: list[int]
    diagnostics: dict
    metadata: dict
    validation: list[int] = field(default_factory=list)
    findings: list = field(default_factory=list)

    def assignments(self):
        """Return partition labels in original row order."""
        labels = ["excluded"] * (len(self.train) + len(self.validation) + len(self.test) + len(self.excluded))
        for label, rows in (("train", self.train), ("validation", self.validation), ("test", self.test)):
            for row in rows:
                labels[row] = label
        return labels


def _components(n, key_sets, edges=()):
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
    for i, j in edges:
        parent[root(i)] = root(j)
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
    """Generate explicit partitions and their scientific diagnostics without mutating input."""
    from .split_engine import run_split
    return run_split(data, config)
