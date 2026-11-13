# memory.py — Structured Causal & Geometric Experience Memory for Closed-Loop Calibration
from dataclasses import dataclass, field, asdict
import json
import os


FAILURE_TAXONOMY = {
    "F_ORDER": "Topological / Base-Grasp Order Violation",
    "F_XY_ALIGN": "Planar XY Offset or Relational Geometry Error",
    "F_Z_DROP": "Vertical Release Height / Impact Topple Error",
    "F_COLLISION": "Swept-Volume / Obstacle or Tower Transit Collision",
    "F_CONTAINER_RIM": "Container Rim Collision or Cavity Offset Error",
    "F_OCCLUSION_STALE": "Occlusion / Stale Pre-Action Coordinate Reuse",
    "F_EXEC_SYNTAX": "Malformed or Unsafe Policy Action Sequence",
}

DEFAULT_KINEMATIC_PRIORS = {
    "transit_z": 0.56,          # Naive uncalibrated transit Z (will collide with 11cm obstacle or tall stack)
    "release_margin_z": 0.075,  # Naive uncalibrated drop height (slightly high -> bounce risk on 4-stack)
    "row_spacing": 0.062,
    "diagonal_delta": 0.062,
    "grasp_force": 75.0,
    "require_home_relocate": False,
}


@dataclass
class FailureDiagnostic:
    failure_code: str
    summary: str
    param_updates: dict = field(default_factory=dict)
    symbolic_rule: str = ""


class CalibrationMemory:
    """
    Stores deduplicated causal failure diagnostics and quantitative kinematic priors
    keyed by canonical task family. Eliminates verbose duplicate prompt text bloat.
    """

    def __init__(self, storage_dir: str = "calibrations"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def _path(self, canonical_type: str) -> str:
        slug = canonical_type.lower()
        return os.path.join(self.storage_dir, f"structured_{slug}.json")

    def load_entry(self, canonical_type: str) -> dict:
        path = self._path(canonical_type)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {
            "canonical_type": canonical_type,
            "calibrated": False,
            "attempts": 0,
            "success_count": 0,
            "kinematic_priors": dict(DEFAULT_KINEMATIC_PRIORS),
            "rules_by_code": {},
            "history": [],
        }

    def save_entry(self, canonical_type: str, data: dict):
        path = self._path(canonical_type)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def record_failure(self, canonical_type: str, diag: FailureDiagnostic):
        data = self.load_entry(canonical_type)
        data["attempts"] = data.get("attempts", 0) + 1
        code = diag.failure_code if diag.failure_code in FAILURE_TAXONOMY else "F_XY_ALIGN"

        priors = data.setdefault("kinematic_priors", dict(DEFAULT_KINEMATIC_PRIORS))
        for k, v in diag.param_updates.items():
            if k == "transit_z":
                priors[k] = max(float(priors.get(k, 0.56)), float(v))
            elif k == "release_margin_z":
                priors[k] = float(v)
            else:
                priors[k] = v

        if diag.symbolic_rule:
            data.setdefault("rules_by_code", {})[code] = diag.symbolic_rule

        data.setdefault("history", []).append(asdict(diag))
        self.save_entry(canonical_type, data)

    def mark_calibrated(self, canonical_type: str, raw_task: str):
        data = self.load_entry(canonical_type)
        data["calibrated"] = True
        data["last_task"] = raw_task
        data["success_count"] = data.get("success_count", 0) + 1
        self.save_entry(canonical_type, data)

    def is_calibrated(self, canonical_type: str) -> bool:
        return bool(self.load_entry(canonical_type).get("calibrated", False))

    def get_kinematic_priors(self, canonical_type: str) -> dict:
        entry = self.load_entry(canonical_type)
        merged = dict(DEFAULT_KINEMATIC_PRIORS)
        merged.update(entry.get("kinematic_priors", {}))
        return merged

    def get_symbolic_rules(self, canonical_type: str) -> list[str]:
        entry = self.load_entry(canonical_type)
        rules_map = entry.get("rules_by_code", {})
        return [r for r in rules_map.values() if r]

