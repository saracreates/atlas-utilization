"""
Apply parsing-stage event filters from YAML (particle count ranges + kinematic cuts).

Maps YAML keys (e.g. ``electrons``) to awkward record fields (``Electrons``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import awkward as ak
import numpy as np

from services.calculations import physics_calcs

YAML_PARTICLE_KEYS: Dict[str, str] = {
    "electrons": "Electrons",
    "muons": "Muons",
    "jets": "Jets",
    "bjets": "BJets",
    "photons": "Photons",
    "taus": "Taus",
}


def canonical_particle_field_name(key: str) -> str:
    return YAML_PARTICLE_KEYS.get(key.lower(), key)


def normalize_yaml_kinematic_cuts(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Turn ``pt_min`` / ``eta_max`` / ``rel_isolation_max`` into internal cut dict."""
    out: Dict[str, Any] = {}
    if "pt" in raw and isinstance(raw["pt"], dict):
        out["pt"] = dict(raw["pt"])
    elif "pt_min" in raw:
        out["pt"] = {"min": float(raw["pt_min"])}

    if "eta" in raw and isinstance(raw["eta"], dict):
        out["eta"] = dict(raw["eta"])
    elif "eta_max" in raw:
        em = float(raw["eta_max"])
        out["eta"] = {"min": -em, "max": em}

    if "phi" in raw and isinstance(raw["phi"], dict):
        out["phi"] = dict(raw["phi"])
    elif "phi_min" in raw or "phi_max" in raw:
        out["phi"] = {
            "min": float(raw.get("phi_min", -np.pi)),
            "max": float(raw.get("phi_max", np.pi)),
        }

    if "rel_isolation_max" in raw:
        out["rel_isolation_max"] = float(raw["rel_isolation_max"])

    return out


def normalize_yaml_overlap_removal(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map YAML particle keys in each rule's ``keep``/``remove`` to canonical field names."""
    rules: List[Dict[str, Any]] = []
    for rule in raw:
        rules.append({
            "keep": canonical_particle_field_name(rule["keep"]),
            "remove": canonical_particle_field_name(rule["remove"]),
            "delta_r_min": float(rule["delta_r_min"]),
        })
    return rules


def apply_parsing_event_selection(
    events: ak.Array,
    particle_counts: Optional[Dict[str, Any]] = None,
    kinematic_cuts: Optional[Dict[str, Any]] = None,
    overlap_removal: Optional[List[Dict[str, Any]]] = None,
) -> ak.Array:
    """
    Kinematic cuts are applied per particle type first, then delta-R overlap
    removal between collections, then event-level particle count ranges —
    count ranges must see post-overlap-removal counts, not raw ones.
    """
    if kinematic_cuts:
        by_obj: Dict[str, Dict[str, Any]] = {}
        for key, val in kinematic_cuts.items():
            if not isinstance(val, dict):
                continue
            cname = canonical_particle_field_name(key)
            by_obj[cname] = normalize_yaml_kinematic_cuts(val)
        events = physics_calcs.filter_events_by_kinematics(events, by_obj)

    if overlap_removal:
        rules = normalize_yaml_overlap_removal(overlap_removal)
        events = physics_calcs.remove_overlapping_objects(events, rules)

    if particle_counts:
        mapped: Dict[str, Any] = {}
        for key, val in particle_counts.items():
            cname = canonical_particle_field_name(key)
            mapped[cname] = val
        events = physics_calcs.filter_events_by_particle_counts(
            events,
            mapped,
            is_exact_count=False,
            is_particle_counts_range=True,
        )

    return events
