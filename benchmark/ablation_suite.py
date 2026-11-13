# ablation_suite.py — Shared Tasks, Parallel Seed Worker, Component Leave-One-Out & Domain Randomization Suite
import json
import math
import multiprocessing
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from simulation.env import RoboTeachEnv
from simulation.camera import SimCamera
from perception.vision_locator import VisionLocator
from skills.primitives import RobotSkills
from calibration.ontology import CanonicalTaskOntology
from calibration.memory import DEFAULT_KINEMATIC_PRIORS
from evaluation.oracle import GroundTruthOracle
from evaluation.stats import wilson_ci, mcnemar_exact, percentile_summary
from planner.progprompt import ProgPrompt
from planner.cap import CaP

BENCHMARK_TASKS = [
    {
        "id": "T1_STACK_ORDERED",
        "family": "STACK_ORDERED",
        "command_tr": "kutuları kule gibi üst üste diz: en altta mavi, üstünde kırmızı, üstünde yeşil, en üstte sarı",
        "synonym_tr": "mavi, kırmızı, yeşil ve sarı küpleri sırasıyla üst üste istifle",
    },
    {
        "id": "T2_ALIGN_ROW",
        "family": "ALIGN_ROW",
        "command_tr": "kırmızı, mavi, yeşil ve sarı küpleri yan yana tek sıra halinde diz",
        "synonym_tr": "tüm küpleri yan yana hizala: kırmızı, mavi, yeşil, sarı",
    },
    {
        "id": "T3_DIAGONAL_PAIR",
        "family": "DIAGONAL_PAIR",
        "command_tr": "kırmızının çaprazına mavi kutuyu, sarının çaprazına yeşil kutuyu koy",
        "synonym_tr": "kırmızı küpün çaprazına mavi küpü ve sarı küpün çaprazına yeşil küpü yerleştir",
    },
    {
        "id": "T4_PLACE_INSIDE",
        "family": "PLACE_INSIDE",
        "command_tr": "kırmızı, mavi ve yeşil küpleri kahverengi kutunun içine koy",
        "synonym_tr": "kırmızı, mavi ve yeşil blokları açık kutunun içine yerleştir",
    },
    {
        "id": "T5_UNSTACK_SWAP",
        "family": "UNSTACK_SWAP",
        "command_tr": "mavi küpü kırmızının üstüne koy ama kırmızı zaten mavinin üstünde",
        "synonym_tr": "maviyi kırmızı küpün üzerine istifle fakat kırmızı zaten mavinin üzerinde",
    },
    {
        "id": "T6_OBSTACLE_CLEARANCE",
        "family": "OBSTACLE_CLEARANCE",
        "command_tr": "engelin üzerinden aşırarak mavi küpü kırmızı küpün üstüne koy",
        "synonym_tr": "duvar engelini aşarak mavi bloğu kırmızının üzerine yerleştir",
    },
]


def naive_single_view_locate(camera: SimCamera, locator: VisionLocator, object_name: str, frame: tuple | None = None) -> list[float] | None:
    """Overhead-only pixel centroid + single-pixel depth lookup without multi-view ray compensation."""
    rgb, depth, vm, _ = frame if frame is not None else camera.capture_rgbd("top")
    mask = locator._segment_chromatic(rgb, object_name) & (depth < 0.992)
    ys, xs = np.where(mask)
    if len(ys) < 15:
        return None
    cy, cx = int(np.mean(ys)), int(np.mean(xs))
    d = float(depth[cy, cx])
    wp = camera.pixel_to_world(cx, cy, d, vm)
    if "cube" in object_name:
        wp[2] -= 0.025
    return wp


class SingleViewLocatorAdapter:
    """Wraps overhead-only single-pixel localization for single-view baselines/ablations."""

    def __init__(self, camera: SimCamera, base_locator: VisionLocator):
        self.camera = camera
        self.base = base_locator
        self._last_metadata = {}

    def clear_cache(self):
        self._last_metadata = {}

    def locate(self, object_name: str) -> list[float] | None:
        return naive_single_view_locate(self.camera, self.base, object_name)

    def locate_all_detailed(self) -> dict[str, dict]:
        frame = self.camera.capture_rgbd("top")
        out = {}
        for name in self.base.object_dimensions:
            pos = naive_single_view_locate(self.camera, self.base, name, frame=frame)
            if pos is not None:
                out[name] = {"position": pos, "z_top": pos[2] + 0.025, "occluded_from_top": False}
        self._last_metadata = out
        return out


def describe_perceived_scene(det: dict[str, dict]) -> str:
    parts = []
    for name, info in det.items():
        p3 = [round(v, 3) for v in info["position"]]
        occ = " (occluded overhead, fused via lateral views)" if info.get("occluded_from_top") else ""
        parts.append(f"{name} at {p3}{occ}")
    return "Perceived 3D workspace: " + "; ".join(parts)


def collect_scene(loc) -> tuple[list[dict], str, dict]:
    det = loc.locate_all_detailed()
    objs = [
        {"name": k, "position": v["position"], "z_top": v.get("z_top", v["position"][2] + 0.025)}
        for k, v in det.items()
    ]
    return objs, describe_perceived_scene(det), det


def freeform_llm_task_key(client, task: str, model: str = "gpt-6-luna") -> str:
    """Reproduces unstructured free-form LLM snake_case task naming (to measure synonym split-brain)."""
    if client is None:
        return "generic_task"
    for attempt in range(4):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Invent a short 2-3 word snake_case identifier for this robot manipulation command. "
                            f"Respond with ONLY the snake_case name.\nCommand: {task}"
                        ),
                    }
                ],
            }
            if "gpt-6" in model or "gpt-5" in model:
                kwargs["reasoning_effort"] = "none"
                kwargs["max_completion_tokens"] = 80
                kwargs["temperature"] = 0.2
            else:
                kwargs["reasoning_effort"] = "low"
                kwargs["max_tokens"] = 90
                kwargs["temperature"] = 0.2
            resp = client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            if "</think>" in raw:
                raw = raw.split("</think>")[-1].strip()
            cleaned = re.sub(r"[^a-z0-9_]", "_", raw.lower()).strip("_")
            if cleaned:
                return cleaned
        except Exception:
            time.sleep(3 * (attempt + 1))
    return "generic_task"


def eval_seed_worker(cfg: dict) -> list[dict]:
    """
    Isolated process worker running one evaluation tier on a slice of unseen seeds.
    Each process initializes its own headless PyBullet client + ProgPrompt client.
    """
    seeds = cfg["seeds"]
    cmd = cfg["cmd"]
    use_multiview = bool(cfg.get("use_multiview", True))
    use_ontology = bool(cfg.get("use_ontology", True))
    use_checkpoints = bool(cfg.get("use_checkpoints", True))
    priors = dict(cfg.get("priors") or DEFAULT_KINEMATIC_PRIORS)
    priors["auto_safe_lift"] = bool(cfg.get("auto_safe_lift", False))
    rules = cfg.get("rules")
    domain_rand = bool(cfg.get("domain_randomization", False))

    env = RoboTeachEnv(headless=True, seed=500, include_container=True, include_obstacle=False)
    camera = SimCamera(headless=True, physics_client_id=env.client)
    mv_locator = VisionLocator(camera, env.object_dimensions)
    active_loc = mv_locator if use_multiview else SingleViewLocatorAdapter(camera, mv_locator)
    skills = RobotSkills(env.robot_id, env.objects, active_loc, control_mode="pd", realtime=False, physics_client_id=env.client)
    ont = CanonicalTaskOntology(llm_client=None)
    oracle = GroundTruthOracle(env)
    planner = ProgPrompt()
    cap = CaP(skills, physics_client_id=env.client)

    spec = ont.parse(cfg["canonical_cmd"])
    eval_spec = ont.parse(cmd)
    scene_cfg = spec.to_scene_config()
    checkpoints = ont.decompose_checkpoints(eval_spec) if (use_ontology and use_checkpoints) else [cmd]

    records = []
    try:
        for seed in seeds:
            if domain_rand:
                camera.set_perturbation(enabled=True, seed=seed, depth_noise_mm=3.0, extrinsics_mm=3.0, extrinsics_deg=1.5)
            else:
                camera.set_perturbation(enabled=False)

            env.reset()
            env.randomize(scene_cfg, seed=seed, domain_randomization=domain_rand)
            skills.home()
            skills.configure_params(priors)

            ok_all = True
            fail_code = "None"
            for cp in checkpoints:
                active_loc.clear_cache()
                objs, desc, _ = collect_scene(active_loc)
                plan = planner.plan(
                    task=cp,
                    scene_description=desc,
                    objects=objs,
                    extra_instructions=rules,
                    kinematic_priors=priors,
                    canonical_type=eval_spec.canonical_type if use_ontology else None,
                    task_spec=(eval_spec if (use_ontology and len(checkpoints) == 1) else None),
                )
                if not plan or not cap.execute(plan, scorer=None, enable_step_integrity=False):
                    ok_all = False
                    fail_code = cap.last_failure_code or "F_EXEC_SYNTAX"
                    skills.home()
                    break
                skills.home()

            gt = oracle.evaluate(spec)
            success = bool(ok_all and gt["success"])
            if not success and fail_code == "None":
                fail_code = gt.get("failure_code", "F_XY_ALIGN")
            records.append({
                "seed": seed,
                "success": success,
                "score": float(gt["score"]),
                "failure_code": fail_code,
            })
    finally:
        env.close()

    stats = planner.client.get_stats() if hasattr(planner.client, "get_stats") else {"call_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "latencies_ms": []}
    return {"records": records, "llm_stats": stats, "model": planner.model}


def run_tier_parallel(
    cmd: str,
    canonical_cmd: str,
    seeds: list[int],
    workers: int = 4,
    **tier_opts,
) -> dict:
    """Dispatches a tier's seed list across headless PyBullet OS processes and aggregates CI metrics."""
    if workers <= 1 or len(seeds) <= 2:
        res = eval_seed_worker({"seeds": seeds, "cmd": cmd, "canonical_cmd": canonical_cmd, **tier_opts})
        recs = res["records"]
        all_stats = [res["llm_stats"]]
        model_used = res["model"]
    else:
        chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
        all_recs = []
        all_stats = []
        model_used = ""
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as pool:
            cfgs = [{"seeds": ch, "cmd": cmd, "canonical_cmd": canonical_cmd, **tier_opts} for ch in chunks]
            for res in pool.map(eval_seed_worker, cfgs):
                all_recs.extend(res["records"])
                all_stats.append(res["llm_stats"])
                if res.get("model"):
                    model_used = res["model"]
        by_seed = {r["seed"]: r for r in all_recs}
        recs = [by_seed[s] for s in seeds]

    outcomes = [bool(r["success"]) for r in recs]
    succ = sum(outcomes)
    n = len(outcomes)
    ci_lo, ci_hi = wilson_ci(succ, n)
    fail_hist: dict[str, int] = {}
    for r in recs:
        if not r["success"]:
            fc = r.get("failure_code") or "F_XY_ALIGN"
            fail_hist[fc] = fail_hist.get(fc, 0) + 1
    lat_ms = [lat for st in all_stats for lat in st.get("latencies_ms", [])]
    return {
        "success": succ,
        "trials": n,
        "sr_pct": round(100.0 * succ / max(1, n), 2),
        "ci95_pct": [ci_lo, ci_hi],
        "seed_outcomes": outcomes,
        "failure_distribution": fail_hist,
        "model": model_used,
        "llm_usage": {
            "calls": sum(st.get("call_count", 0) for st in all_stats),
            "prompt_tokens": sum(st.get("prompt_tokens", 0) for st in all_stats),
            "completion_tokens": sum(st.get("completion_tokens", 0) for st in all_stats),
            "latency_ms": percentile_summary(lat_ms) if lat_ms else {},
        },
    }


def summarize_tier_across_tasks(per_task: dict) -> dict:
    tot_s = sum(v["success"] for v in per_task.values())
    tot_n = sum(v["trials"] for v in per_task.values())
    ci_lo, ci_hi = wilson_ci(tot_s, tot_n)
    matched: list[bool] = []
    for item in BENCHMARK_TASKS:
        tid = item["id"]
        if tid in per_task:
            matched.extend(per_task[tid].get("seed_outcomes", []))
    return {
        "total_success": tot_s,
        "total_trials": tot_n,
        "overall_sr_pct": round(100.0 * tot_s / max(1, tot_n), 2),
        "ci95_pct": [ci_lo, ci_hi],
        "matched_outcomes": matched,
    }


def run_component_and_dr_ablation(
    trials_per_task: int = 30,
    calib_info: dict | None = None,
    out_path: str | None = None,
    task_filter: set[str] | None = None,
    workers: int = 4,
) -> dict:
    """
    Runs leave-one-out component ablations from Full K4 plus Domain Randomization (DR)
    robustness checks against Fair CaP Multi-View across matched unseen seeds.
    """
    curves = dict(calib_info or {})
    methods = {
        "ablation_no_quant_priors": {},
        "ablation_no_symbolic_rules": {},
        "ablation_no_multiview": {},
        "ablation_no_ontology": {},
        "dr_fair_cap_multiview": {},
        "dr_neuro_symbolic_roboteach": {},
    }
    if task_filter and out_path and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as fh:
            prev = json.load(fh)
        for m in methods:
            methods[m].update(prev.get("results_by_method", {}).get(m, {}))

    for t_idx, item in enumerate(BENCHMARK_TASKS):
        tid = item["id"]
        if task_filter and tid not in task_filter:
            continue
        cmd = item["command_tr"]
        syn_cmd = item["synonym_tr"]
        seeds = [3000 + t_idx * 100 + tr for tr in range(trials_per_task)]
        c_info = curves.get(tid, {})
        learned_rules = c_info.get("learned_rules") or None
        final_priors = dict(c_info.get("final_priors") or DEFAULT_KINEMATIC_PRIORS)

        methods["ablation_no_quant_priors"][tid] = run_tier_parallel(
            cmd=syn_cmd, canonical_cmd=cmd, seeds=seeds, workers=workers,
            use_multiview=True, use_ontology=True, use_checkpoints=True,
            priors=dict(DEFAULT_KINEMATIC_PRIORS), auto_safe_lift=False, rules=learned_rules,
        )
        methods["ablation_no_symbolic_rules"][tid] = run_tier_parallel(
            cmd=syn_cmd, canonical_cmd=cmd, seeds=seeds, workers=workers,
            use_multiview=True, use_ontology=True, use_checkpoints=True,
            priors=final_priors, auto_safe_lift=True, rules=None,
        )
        methods["ablation_no_multiview"][tid] = run_tier_parallel(
            cmd=syn_cmd, canonical_cmd=cmd, seeds=seeds, workers=workers,
            use_multiview=False, use_ontology=True, use_checkpoints=True,
            priors=final_priors, auto_safe_lift=True, rules=learned_rules,
        )
        methods["ablation_no_ontology"][tid] = run_tier_parallel(
            cmd=syn_cmd, canonical_cmd=cmd, seeds=seeds, workers=workers,
            use_multiview=True, use_ontology=False, use_checkpoints=False,
            priors=final_priors, auto_safe_lift=True, rules=learned_rules,
        )
        methods["dr_fair_cap_multiview"][tid] = run_tier_parallel(
            cmd=cmd, canonical_cmd=cmd, seeds=seeds, workers=workers,
            use_multiview=True, use_ontology=False, use_checkpoints=False,
            priors=dict(DEFAULT_KINEMATIC_PRIORS), auto_safe_lift=False, rules=None,
            domain_randomization=True,
        )
        methods["dr_neuro_symbolic_roboteach"][tid] = run_tier_parallel(
            cmd=syn_cmd, canonical_cmd=cmd, seeds=seeds, workers=workers,
            use_multiview=True, use_ontology=True, use_checkpoints=True,
            priors=final_priors, auto_safe_lift=True, rules=learned_rules,
            domain_randomization=True,
        )
        print(
            f"[Comp+DR {tid}] -Priors={methods['ablation_no_quant_priors'][tid]['success']}/{trials_per_task} | "
            f"-Rules={methods['ablation_no_symbolic_rules'][tid]['success']}/{trials_per_task} | "
            f"-MV={methods['ablation_no_multiview'][tid]['success']}/{trials_per_task} | "
            f"-Ont={methods['ablation_no_ontology'][tid]['success']}/{trials_per_task} | "
            f"DR_CaP={methods['dr_fair_cap_multiview'][tid]['success']}/{trials_per_task} vs "
            f"DR_NS={methods['dr_neuro_symbolic_roboteach'][tid]['success']}/{trials_per_task}",
            flush=True,
        )
        if out_path:
            payload = _build_component_payload(trials_per_task, methods)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)

    return _build_component_payload(trials_per_task, methods)


def _build_component_payload(trials_per_task: int, methods: dict) -> dict:
    summary = {}
    matched_map = {}
    for m, per_task in methods.items():
        s = summarize_tier_across_tasks(per_task)
        matched_map[m] = s.pop("matched_outcomes")
        summary[m] = s
    dr_cap = matched_map.get("dr_fair_cap_multiview", [])
    dr_ns = matched_map.get("dr_neuro_symbolic_roboteach", [])
    mcnemar_dr = mcnemar_exact(dr_ns, dr_cap) if dr_cap and len(dr_cap) == len(dr_ns) else {}
    return {
        "trials_per_task": trials_per_task,
        "domain_randomization_spec": {
            "mass_range_kg": [0.084, 0.156],
            "cube_friction_range": [1.2, 2.8],
            "table_friction_range": [1.0, 2.2],
            "landmark_xy_jitter_m": 0.008,
            "depth_noise_mm": 3.0,
            "extrinsics_mm": 3.0,
            "extrinsics_deg": 1.5,
        },
        "results_by_method": methods,
        "summary": summary,
        "mcnemar_dr_ns_vs_cap": mcnemar_dr,
    }

