"""Opt-in domain presets with role-to-column mapping and an external entry-point API."""

from importlib.metadata import entry_points

BUILTINS = {
    "batteries": {"temperature_k": [0, None], "capacity": [0, None], "cycle": [0, None], "salt_concentration": [0, None]},
    "molecular": {"molecular_weight": [0, None]},
    "materials": {"density": [0, None], "porosity_fraction": [0, 1], "temperature_k": [0, None]},
    "process": {"pressure_absolute": [0, None], "temperature_k": [0, None], "mass_flow_magnitude": [0, None]},
}


def available_plugins():
    return {"builtin": sorted(BUILTINS), "external": sorted(e.name for e in entry_points(group="chemdata_auditor.presets"))}


def preset(name, columns):
    """Return an explicit AuditConfig; only supplied roles activate constraints."""
    from .audit import AuditConfig
    if not isinstance(columns, dict) or not columns or any(not isinstance(v, str) or not v for v in columns.values()):
        raise ValueError("Preset columns must map one or more semantic roles to column names.")
    if name not in BUILTINS:
        selected = [e for e in entry_points(group="chemdata_auditor.presets") if e.name == name]
        if len(selected) != 1:
            raise ValueError(f"Unknown or ambiguous preset: {name}")
        result = selected[0].load()(dict(columns))
        if not isinstance(result, AuditConfig):
            raise ValueError("External preset must return AuditConfig.")
        result.__post_init__()
        return result
    common = {"sample_id", "source", "laboratory", "measured_at"}
    if name == "molecular":
        common |= {"smiles"}
    unknown = set(columns) - set(BUILTINS[name]) - common
    if unknown:
        raise ValueError(f"Unknown roles for {name}: {sorted(unknown)}")
    return AuditConfig(
        bounds={columns[k]: bounds for k, bounds in BUILTINS[name].items() if k in columns},
        provenance_columns=[columns[k] for k in ("source", "laboratory", "measured_at") if k in columns],
        identifier_columns=[columns["sample_id"]] if "sample_id" in columns else [],
        group_columns=[columns["sample_id"]] if "sample_id" in columns else [],
        smiles_column=columns.get("smiles") if name == "molecular" else None,
    )
