"""Dataset auditing and scientific splitting with explicit assumptions."""

from .audit import AuditConfig, AuditReport, Finding, audit
from .split import SplitConfig, SplitResult, split

__version__ = "0.2.0"
__all__ = ["AuditConfig", "AuditReport", "Finding", "audit", "SplitConfig", "SplitResult", "split"]
