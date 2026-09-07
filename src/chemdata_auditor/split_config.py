"""Validated configuration for scientific two- and three-way evaluation designs."""

from dataclasses import dataclass, field
import re

from .common import finite_number, names
from .diagnostics import DiagnosticsConfig

STRATEGIES = {"random", "formulation", "composition", "scaffold", "publication", "laboratory", "time", "temporal", "cluster", "extrapolation"}


@dataclass
class SplitConfig:
    strategy: str
    columns: list[str] = field(default_factory=list)
    group_columns: list[str] = field(default_factory=list)
    test_size: float = 0.2
    validation_size: float = 0.0
    seed: int = 0
    holdout_values: list | None = None
    validation_values: list | None = None
    cutoff: str | None = None
    validation_cutoff: str | None = None
    threshold: float | None = None
    validation_threshold: float | None = None
    direction: str = "high"
    extrapolation_bounds: dict[str, list[float]] = field(default_factory=dict)
    validation_bounds: dict[str, list[float]] = field(default_factory=dict)
    crossing_policy: str = "error"
    composition_decimals: int | None = None
    n_clusters: int = 5
    cluster_scales: dict[str, float] = field(default_factory=dict)
    group_near_duplicates: dict[str, float] = field(default_factory=dict)
    group_smiles_column: str | None = None
    group_molecular_similarity: float | None = None
    target_column: str | None = None
    allow_target_in_split: bool = False
    max_pair_comparisons: int = 2000000
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError("Unknown split strategy.")
        for key in ("columns", "group_columns"):
            names(getattr(self, key), key)
        if self.strategy == "random":
            if self.columns:
                raise ValueError("Random allocation uses no scientific columns; use group_columns for grouping constraints.")
        elif not self.columns:
            raise ValueError("A scientific split requires columns.")
        if self.strategy in {"formulation", "publication", "laboratory", "scaffold", "time", "temporal"} and len(self.columns) != 1:
            raise ValueError("This strategy requires exactly one scientific column.")
        finite_number(self.test_size, "test_size")
        finite_number(self.validation_size, "validation_size")
        if not 0 < self.test_size < 1 or not 0 <= self.validation_size < 1 or self.test_size + self.validation_size >= 1:
            raise ValueError("Partition fractions must leave nonempty training mass; validation_size may be zero.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if self.direction not in {"high", "low"} or self.crossing_policy not in {"error", "exclude"}:
            raise ValueError("direction must be high/low and crossing_policy error/exclude.")
        is_time = self.strategy in {"time", "temporal"}
        if is_time:
            if self.cutoff is None:
                raise ValueError("Time splits require an explicit cutoff.")
            for value in (self.cutoff, self.validation_cutoff):
                if value is not None and (not isinstance(value, str) or not re.match(r"^\d{4}-\d{2}-\d{2}(?:T| |$)", value)):
                    raise ValueError("Time cutoffs must be absolute ISO dates/timestamps.")
        elif self.cutoff is not None or self.validation_cutoff is not None:
            raise ValueError("Time cutoffs only apply to temporal splits.")
        for key in ("extrapolation_bounds", "validation_bounds", "cluster_scales", "group_near_duplicates", "diagnostics"):
            if not isinstance(getattr(self, key), dict):
                raise ValueError(f"{key} must be an object.")
        if self.strategy == "extrapolation":
            if bool(self.extrapolation_bounds) == (self.threshold is not None):
                raise ValueError("Use exactly one threshold or extrapolation_bounds.")
            if self.threshold is not None:
                finite_number(self.threshold, "threshold")
                if len(self.columns) != 1 or self.validation_bounds:
                    raise ValueError("Threshold extrapolation requires one column and no validation_bounds.")
                if self.validation_threshold is not None:
                    finite_number(self.validation_threshold, "validation_threshold")
                    if (self.direction == "high" and self.validation_threshold >= self.threshold) or (self.direction == "low" and self.validation_threshold <= self.threshold):
                        raise ValueError("Validation threshold must lie inside the development range.")
            else:
                if self.validation_threshold is not None or self.direction != "high":
                    raise ValueError("Box extrapolation does not use threshold or direction options.")
                for key in ("extrapolation_bounds", "validation_bounds"):
                    bounds = getattr(self, key)
                    if not bounds and key == "validation_bounds":
                        continue
                    if set(bounds) != set(self.columns):
                        raise ValueError("Bounds must specify every extrapolation column.")
                    for limits in bounds.values():
                        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
                            raise ValueError("Bounds require [lower, upper].")
                        for v in limits:
                            finite_number(v, "extrapolation bound")
                        if limits[0] > limits[1]:
                            raise ValueError("Bounds are reversed.")
                for c, (lower, upper) in self.validation_bounds.items():
                    outer = self.extrapolation_bounds[c]
                    if lower < outer[0] or upper > outer[1]:
                        raise ValueError("validation_bounds must be contained in extrapolation_bounds.")
        elif self.threshold is not None or self.validation_threshold is not None or self.extrapolation_bounds or self.validation_bounds or self.direction != "high":
            raise ValueError("Extrapolation settings only apply to extrapolation splits.")
        for key in ("holdout_values", "validation_values"):
            values = getattr(self, key)
            if values is not None:
                if self.strategy not in {"formulation", "publication", "laboratory"} or not isinstance(values, list) or not values:
                    raise ValueError("Explicit values need a nonempty list for formulation, publication, or laboratory holdout.")
                for v in values:
                    if isinstance(v, bool) or not isinstance(v, (str, int, float)):
                        raise ValueError("Holdout labels must be strings or finite numbers.")
                    if isinstance(v, (int, float)):
                        finite_number(v, key)
                if len(set(values)) != len(values):
                    raise ValueError("Holdout labels cannot repeat.")
        if self.validation_values is not None and (self.holdout_values is None or set(self.validation_values) & set(self.holdout_values)):
            raise ValueError("Validation values need disjoint explicit test holdout_values.")
        if is_time or self.strategy == "extrapolation" or self.holdout_values is not None:
            if self.test_size != 0.2 or self.validation_size != 0 or self.seed != 0:
                raise ValueError("Fractions and seeds do not apply to explicit value/boundary allocation; use explicit validation boundaries/values.")
        if not is_time and self.strategy != "extrapolation" and self.crossing_policy != "error":
            raise ValueError("crossing_policy only applies to time/extrapolation boundaries.")
        if self.composition_decimals is not None:
            if self.strategy != "composition" or isinstance(self.composition_decimals, bool) or not isinstance(self.composition_decimals, int) or not 0 <= self.composition_decimals <= 15:
                raise ValueError("composition_decimals requires composition and an integer in [0,15].")
        if isinstance(self.n_clusters, bool) or not isinstance(self.n_clusters, int) or self.n_clusters < 2:
            raise ValueError("n_clusters must be an integer of at least two.")
        if self.strategy != "cluster" and (self.n_clusters != 5 or self.cluster_scales):
            raise ValueError("Cluster options only apply to cluster holdouts.")
        if self.strategy == "cluster" and set(self.cluster_scales) != set(self.columns):
            raise ValueError("Cluster holdouts require an explicit scale for every clustering column.")
        for key in ("cluster_scales", "group_near_duplicates"):
            names(list(getattr(self, key)), key)
            for v in getattr(self, key).values():
                finite_number(v, key)
                if v < 0 or (key == "cluster_scales" and v == 0):
                    raise ValueError("Scales must be positive and tolerances nonnegative.")
        if self.group_smiles_column is not None and (not isinstance(self.group_smiles_column, str) or not self.group_smiles_column):
            raise ValueError("group_smiles_column must be a name.")
        if self.group_molecular_similarity is not None:
            finite_number(self.group_molecular_similarity, "group_molecular_similarity")
            if not self.group_smiles_column or not 0 < self.group_molecular_similarity <= 1:
                raise ValueError("Molecular grouping needs a SMILES column and threshold in (0,1].")
        if self.target_column is not None and (not isinstance(self.target_column, str) or not self.target_column):
            raise ValueError("target_column must be a name.")
        if not isinstance(self.allow_target_in_split, bool):
            raise ValueError("allow_target_in_split must be boolean.")
        if self.target_column and self.target_column in {*self.columns, *self.group_columns, *self.group_near_duplicates} and not self.allow_target_in_split:
            raise ValueError("Target-dependent partition design requires allow_target_in_split=true and explicit disclosure.")
        if isinstance(self.max_pair_comparisons, bool) or not isinstance(self.max_pair_comparisons, int) or self.max_pair_comparisons < 1:
            raise ValueError("max_pair_comparisons must be a positive integer.")
        DiagnosticsConfig(**self.diagnostics)
