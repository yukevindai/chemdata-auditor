"""Explicit, bounded similarity calculations shared by audit and split diagnostics."""

import numpy as np

from .common import numeric, require


def matrix(data, columns):
    require(data, columns)
    return np.column_stack([numeric(data[c]) for c in columns])


def pair_budget(n, limit, m=None):
    count = n * (n - 1) // 2 if m is None else n * m
    if count > limit:
        raise ValueError(f"Similarity check needs {count} comparisons, exceeding max_pair_comparisons={limit}. "
                         "Choose a smaller scientifically justified dataset or explicitly raise the budget.")


def numeric_pairs(values, tolerances, limit):
    """Yield positional pairs within every absolute column tolerance; no scaling fitted."""
    pair_budget(len(values), limit)
    finite = np.isfinite(values).all(axis=1)
    tolerance = np.asarray(tolerances)
    for i in range(len(values) - 1):
        if not finite[i]:
            continue
        match = finite[i + 1:] & (np.abs(values[i + 1:] - values[i]) <= tolerance).all(axis=1)
        for offset in np.flatnonzero(match):
            yield i, int(i + 1 + offset)


def molecules(series):
    """Return canonical identities and Morgan fingerprints; invalid rows remain explicit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise ValueError('Molecular checks need pip install "chemdata-auditor[chem]"') from exc
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    canonical, fingerprints, invalid, cache = [], [], [], {}
    for i, value in enumerate(series):
        if not isinstance(value, str) or not value.strip():
            mol = None
        elif value in cache:
            identity, fingerprint = cache[value]
            canonical.append(identity)
            fingerprints.append(fingerprint)
            continue
        else:
            mol = Chem.MolFromSmiles(value)
        if mol is None or mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1:
            invalid.append(i)
            canonical.append(None)
            fingerprints.append(None)
        else:
            identity = Chem.MolToSmiles(mol, isomericSmiles=True)
            fingerprint = generator.GetFingerprint(mol)
            cache[value] = (identity, fingerprint)
            canonical.append(identity)
            fingerprints.append(fingerprint)
    return canonical, fingerprints, invalid


def molecular_pairs(fingerprints, threshold, limit):
    from rdkit import DataStructs
    pair_budget(len(fingerprints), limit)
    for i, fp in enumerate(fingerprints):
        if fp is None:
            continue
        for j in range(i + 1, len(fingerprints)):
            if fingerprints[j] is not None:
                score = DataStructs.TanimotoSimilarity(fp, fingerprints[j])
                if score >= threshold:
                    yield i, j, float(score)
