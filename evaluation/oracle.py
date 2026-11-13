# oracle.py — Deterministic Simulator Ground-Truth Evaluator & VLM-Judge Agreement Metrics
import math
import pybullet as p

from calibration.ontology import TaskSpec


class GroundTruthOracle:
    """
    Computes exact physical task satisfaction strictly from ground-truth simulator poses,
    orientations, and contacts. NEVER accessed by the agent during planning or perception;
    used exclusively for scientific benchmarking and validating VLM Judge reliability.
    """

    TABLE_Z = 0.42

    def __init__(self, env):
        self.env = env

    def _pose(self, name: str) -> tuple[list[float], list[float]]:
        oid = self.env.objects[name]
        pos, orn = p.getBasePositionAndOrientation(oid, physicsClientId=self.env.client)
        return list(pos), list(orn)

    def _tilt_rad(self, orn: list[float]) -> float:
        rot = p.getMatrixFromQuaternion(orn)
        # Local Z axis vertical component rot[8] = cos(tilt)
        cos_t = max(-1.0, min(1.0, abs(rot[8])))
        return math.acos(cos_t)

    def _any_fallen_off_table(self) -> tuple[bool, str | None]:
        for name in ("red_cube", "blue_cube", "green_cube", "yellow_cube"):
            if name not in self.env.objects:
                continue
            pos, _ = self._pose(name)
            if pos[2] < 0.36 or abs(pos[0]) > 0.58 or abs(pos[1]) > 0.58:
                return True, name
        return False, None

    def evaluate(self, spec: TaskSpec) -> dict:
        fallen, fallen_obj = self._any_fallen_off_table()
        if fallen:
            return {
                "success": False,
                "score": 0.0,
                "failure_code": "F_COLLISION",
                "reason": f"{fallen_obj} fell off the workspace table",
            }

        fam = spec.canonical_type
        if fam in ("STACK_ORDERED", "UNSTACK_SWAP", "OBSTACLE_CLEARANCE"):
            return self._eval_stack(spec.ordered_objects)
        if fam == "ALIGN_ROW":
            return self._eval_row(spec.ordered_objects)
        if fam == "DIAGONAL_PAIR":
            return self._eval_diagonal(spec.relational_pairs)
        if fam == "PLACE_INSIDE":
            return self._eval_inside(spec.ordered_objects, spec.target_container or "brown_box")
        return self._eval_stack(spec.ordered_objects)

    def _eval_stack(self, ordered: list[str]) -> dict:
        if len(ordered) < 2:
            return {"success": False, "score": 0.0, "failure_code": "F_ORDER", "reason": "Insufficient stack specification"}

        poses = [self._pose(c) for c in ordered]
        base_pos, base_orn = poses[0]
        if base_pos[2] > 0.475:
            return {
                "success": False,
                "score": 0.2,
                "failure_code": "F_ORDER",
                "reason": f"Base object {ordered[0]} is not resting on the table (z={base_pos[2]:.3f}m)",
            }

        satisfied_links = 0
        total_links = len(ordered) - 1
        last_issue = ""
        failure_code = "None"

        for i in range(total_links):
            lower_pos, _ = poses[i]
            upper_pos, upper_orn = poses[i + 1]
            xy_err = math.hypot(upper_pos[0] - lower_pos[0], upper_pos[1] - lower_pos[1])
            dz = upper_pos[2] - lower_pos[2]
            tilt = self._tilt_rad(upper_orn)

            if dz < 0.036:
                failure_code = "F_Z_DROP" if xy_err < 0.05 else "F_XY_ALIGN"
                last_issue = f"{ordered[i+1]} is not stacked above {ordered[i]} (dz={dz:.3f}m, xy_err={xy_err:.3f}m)"
                break
            if xy_err > 0.032 or tilt > 0.35:
                failure_code = "F_XY_ALIGN" if xy_err > 0.032 else "F_Z_DROP"
                last_issue = f"{ordered[i+1]} misaligned/tilted on {ordered[i]} (xy_err={xy_err:.3f}m, tilt={tilt:.2f}rad)"
                break
            satisfied_links += 1

        score = satisfied_links / total_links
        success = satisfied_links == total_links
        return {
            "success": success,
            "score": round(score, 3),
            "failure_code": "None" if success else failure_code,
            "reason": "Ordered stack verified" if success else last_issue,
        }

    def _eval_row(self, ordered: list[str]) -> dict:
        poses = [self._pose(c)[0] for c in ordered]
        # All cubes must rest on table
        for c, pos in zip(ordered, poses):
            if abs(pos[2] - 0.445) > 0.025:
                return {
                    "success": False,
                    "score": 0.25,
                    "failure_code": "F_Z_DROP",
                    "reason": f"{c} not resting flat on table (z={pos[2]:.3f}m)",
                }
        xs = [p[0] for p in poses]
        ys = [p[1] for p in poses]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        # Either aligned along Y (small X spread) or along X (small Y spread)
        cross_spread = min(span_x, span_y)
        main_coords = sorted(ys if span_y >= span_x else xs)
        gaps = [main_coords[i + 1] - main_coords[i] for i in range(len(main_coords) - 1)]
        gaps_ok = all(0.046 <= g <= 0.095 for g in gaps)
        collinear = cross_spread <= 0.032

        if collinear and gaps_ok:
            return {"success": True, "score": 1.0, "failure_code": "None", "reason": "Collinear row verified"}
        score = (0.5 if collinear else max(0.0, 0.5 - cross_spread * 3.0)) + (
            0.5 * sum(0.046 <= g <= 0.095 for g in gaps) / max(1, len(gaps))
        )
        return {
            "success": False,
            "score": round(max(0.0, min(0.95, score)), 3),
            "failure_code": "F_XY_ALIGN",
            "reason": f"Row spread={cross_spread:.3f}m, gaps={[round(g,3) for g in gaps]}",
        }

    def _eval_diagonal(self, pairs: list[tuple[str, str]]) -> dict:
        if not pairs:
            return {"success": False, "score": 0.0, "failure_code": "F_XY_ALIGN", "reason": "No diagonal pairs specified"}
        ok_count = 0
        reasons = []
        for mov, ref in pairs:
            mp, _ = self._pose(mov)
            rp, _ = self._pose(ref)
            dx = abs(mp[0] - rp[0])
            dy = abs(mp[1] - rp[1])
            on_table = abs(mp[2] - 0.445) < 0.025 and abs(rp[2] - 0.445) < 0.025
            if on_table and (0.035 <= dx <= 0.115) and (0.035 <= dy <= 0.115):
                ok_count += 1
            else:
                reasons.append(f"({mov},{ref}): dx={dx:.3f}m, dy={dy:.3f}m")
        score = ok_count / len(pairs)
        return {
            "success": ok_count == len(pairs),
            "score": round(score, 3),
            "failure_code": "None" if ok_count == len(pairs) else "F_XY_ALIGN",
            "reason": "Diagonal offsets verified" if ok_count == len(pairs) else "; ".join(reasons),
        }

    def _eval_inside(self, targets: list[str], container_name: str) -> dict:
        if container_name not in self.env.objects:
            return {"success": False, "score": 0.0, "failure_code": "F_CONTAINER_RIM", "reason": "Container missing"}
        box_pos, _ = self._pose(container_name)
        inside_count = 0
        reasons = []
        for c in targets:
            cp, _ = self._pose(c)
            in_xy = abs(cp[0] - box_pos[0]) <= 0.068 and abs(cp[1] - box_pos[1]) <= 0.068
            in_z = 0.425 <= cp[2] <= 0.515
            if in_xy and in_z:
                inside_count += 1
            else:
                reasons.append(f"{c} outside cavity (dx={cp[0]-box_pos[0]:.3f}, dy={cp[1]-box_pos[1]:.3f}, z={cp[2]:.3f})")
        score = inside_count / max(1, len(targets))
        return {
            "success": inside_count == len(targets),
            "score": round(score, 3),
            "failure_code": "None" if inside_count == len(targets) else "F_CONTAINER_RIM",
            "reason": "All target objects inside container" if inside_count == len(targets) else "; ".join(reasons),
        }


def compute_judge_agreement(
    oracle_labels: list[bool],
    vlm_labels: list[bool],
    oracle_scores: list[float] | None = None,
    vlm_scores: list[float] | None = None,
) -> dict:
    """Computes Accuracy, Precision, Recall, F1, Cohen's kappa, and Score MAE."""
    n = len(oracle_labels)
    if n == 0 or len(vlm_labels) != n:
        return {"n": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "cohen_kappa": 0.0}

    tp = sum(1 for o, v in zip(oracle_labels, vlm_labels) if o and v)
    tn = sum(1 for o, v in zip(oracle_labels, vlm_labels) if (not o) and (not v))
    fp = sum(1 for o, v in zip(oracle_labels, vlm_labels) if (not o) and v)
    fn = sum(1 for o, v in zip(oracle_labels, vlm_labels) if o and (not v))

    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    p_yes_o = (tp + fn) / n
    p_yes_v = (tp + fp) / n
    p_e = p_yes_o * p_yes_v + (1.0 - p_yes_o) * (1.0 - p_yes_v)
    kappa = (acc - p_e) / (1.0 - p_e) if abs(1.0 - p_e) > 1e-9 else 1.0

    out = {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "cohen_kappa": round(kappa, 4),
    }
    if oracle_scores and vlm_scores and len(oracle_scores) == n:
        mae = sum(abs(a - b) for a, b in zip(oracle_scores, vlm_scores)) / n
        out["score_mae"] = round(mae, 4)
    return out

