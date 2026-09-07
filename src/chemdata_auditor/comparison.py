"""Compare evaluation designs without ranking them by an invented reliability score."""

from dataclasses import asdict, dataclass, field, replace

from .audit import Finding
from .common import JsonResult, frame, metadata
from .diagnostics import DiagnosticsConfig
from .split import SplitConfig, split


@dataclass
class ComparisonReport(JsonResult):
    results: dict
    metadata: dict
    findings: list = field(default_factory=list)


def compare_splits(data, configurations, diagnostics=None):
    """Return each design, comparable diagnostics, and explicit infeasible-design errors."""
    data = frame(data)
    if not isinstance(configurations, dict) or not configurations or any(not isinstance(name, str) or not name for name in configurations):
        raise ValueError("Provide a nonempty object mapping design names to SplitConfig or configuration objects.")
    if diagnostics is not None:
        diagnostics.__post_init__()
    results, findings, recorded = {}, [], {}
    for name, config in configurations.items():
        try:
            cfg = config if isinstance(config, SplitConfig) else SplitConfig(**config)
            if diagnostics is not None:
                cfg = replace(cfg, diagnostics={**cfg.diagnostics, **asdict(diagnostics)})
            recorded[name] = asdict(cfg)
            result = split(data, cfg)
            evaluation = result.diagnostics["evaluation"]
            summary = {"n_train": len(result.train), "n_validation": len(result.validation), "n_test": len(result.test),
                       "n_excluded": len(result.excluded), "error_findings": sum(f.severity == "error" for f in result.findings),
                       "warning_findings": sum(f.severity == "warning" for f in result.findings),
                       "outside_training_domain_fraction": evaluation["training_domain"].get("outside_fraction"),
                       "generalization": result.diagnostics["generalization"]}
            results[name] = {"status": "ok", "summary": summary, "report": result.to_dict()}
        except (ValueError, TypeError) as exc:
            recorded[name] = asdict(config) if isinstance(config, SplitConfig) else config
            results[name] = {"status": "error", "error": str(exc)}
            findings.append(Finding("infeasible_split_design", "error", [], [], f"{name}: {exc}",
                                    "Revise the scientific design or supply sufficient independent groups; do not silently relax constraints.",
                                    {"design": name}))
    meta = metadata(data, diagnostics or DiagnosticsConfig())
    meta["comparison_configurations"] = recorded
    meta["interpretation"] = "Compare evaluation questions and diagnostics; lower shift or overlap does not prove validity. No model is trained and no performance metric is invented."
    return ComparisonReport(results, meta, findings)
