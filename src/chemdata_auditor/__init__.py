"""Dataset auditing and scientific splitting with explicit assumptions."""

from .audit import AuditConfig, AuditReport, Finding, audit
from .split import SplitConfig, SplitResult, split
from .diagnostics import DiagnosticsConfig, diagnose
from .comparison import ComparisonReport, compare_splits

__version__ = "0.3.0"
__all__ = ["AuditConfig", "AuditReport", "Finding", "audit", "SplitConfig", "SplitResult", "split",
           "DiagnosticsConfig", "diagnose", "ComparisonReport", "compare_splits"]
