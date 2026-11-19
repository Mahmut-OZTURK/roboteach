"""Backup-calibration experiment (safe-Z lift disabled).

Question: when the geometric safety layer (automatic safe-Z lift) is unavailable and the
kinematic priors start at the naive defaults (transit_z = 0.56 m, release_margin_z = 0.075 m),
does the feedback calibration loop recover task success?

Protocol
- Both evaluation arms keep the canonical ontology, sub-goal checkpoints and four-view
  perception enabled; only the calibration memory differs.
- Calibration phase: the K4 calibration loop (gpt-4.1 VLM critic, at most 5 iterations,
  two consecutive VLM approvals) is rerun per task family with auto_safe_lift = False,
  seeds 5000 + 100 * t_idx + it, into a fresh memory directory.
- Evaluation phase: matched unseen seeds 3000 + 100 * t_idx + tr (30 per task, 180 per arm),
  synonym command as in K4.
    arm "default_no_lift":    DEFAULT_KINEMATIC_PRIORS, no rules, auto_safe_lift = False
    arm "calibrated_no_lift": calibrated priors + learned rules, auto_safe_lift = False
- Reference (not rerun): K4 with auto_safe_lift = True from live_ablation_benchmark.json.

Usage
    venv/bin/python benchmark/backup_calibration_experiment.py --trials 30 --workers 4
    # resumes per task if interrupted; restrict with --tasks T6_OBSTACLE_CLEARANCE,T1_STACK_ORDERED
"""
import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"), override=True)

from benchmark.ablation_suite import (  # noqa: E402
    BENCHMARK_TASKS,
    collect_scene,
    run_tier_parallel,
    summarize_tier_across_tasks,
)
from calibration.memory import CalibrationMemory, DEFAULT_KINEMATIC_PRIORS  # noqa: E402
from calibration.ontology import CanonicalTaskOntology  # noqa: E402
from evaluation.oracle import GroundTruthOracle  # noqa: E402
from evaluation.stats import mcnemar_exact, task_stratified_paired_bootstrap_ci, wilson_ci  # noqa: E402
from evaluation.vlm_scorer import VLMScorer  # noqa: E402
from perception.vision_locator import VisionLocator  # noqa: E402
from planner.cap import CaP  # noqa: E402
from planner.progprompt import ProgPrompt  # noqa: E402
from simulation.camera import SimCamera  # noqa: E402
from simulation.env import RoboTeachEnv  # noqa: E402
from skills.primitives import RobotSkills  # noqa: E402

RES_DIR = os.path.join(ROOT, "benchmark", "results")
OUT_PATH = os.path.join(RES_DIR, "backup_calibration_no_safe_lift.json")
MEM_DIR = os.path.join(RES_DIR, "calib_state_no_safe_lift")
ARMS = ("default_no_lift", "calibrated_no_lift")


def _stats_of(client) -> dict:
    if client is not None and hasattr(client, "get_stats"):
        st = client.get_stats()
        return {k: st.get(k, 0) for k in ("call_count", "prompt_tokens", "completion_tokens")}
    return {"call_count": 0, "prompt_tokens": 0, "completion_tokens": 0}


def calibrate_without_safe_lift(t_idx: int, item: dict, max_iters: int) -> dict:
    """K4 calibration loop with the geometric safe-Z lift switched off."""
    env = RoboTeachEnv(headless=True, seed=500, include_container=True, include_obstacle=False)
    camera = SimCamera(headless=True, physics_client_id=env.client)
    loc = VisionLocator(camera, env.object_dimensions)
    skills = RobotSkills(env.robot_id, env.objects, loc, control_mode="pd", realtime=False, physics_client_id=env.client)
    ont = CanonicalTaskOntology(llm_client=None)
    oracle = GroundTruthOracle(env)
    planner = ProgPrompt()
    cap = CaP(skills, physics_client_id=env.client)
    scorer = VLMScorer(camera)

    cmd = item["command_tr"]
    spec = ont.parse(cmd)
    scene_cfg = spec.to_scene_config()

    os.makedirs(MEM_DIR, exist_ok=True)
    mem = CalibrationMemory(storage_dir=MEM_DIR)
    entry_path = mem._path(spec.canonical_type)
    if os.path.exists(entry_path):
        os.remove(entry_path)

    iterations = []
    calibrated = False
    consecutive = 0
    try:
        for it in range(1, max_iters + 1):
            env.reset()
            env.randomize(scene_cfg, seed=5000 + t_idx * 100 + it)
            skills.home()
            priors = mem.get_kinematic_priors(spec.canonical_type)
            priors["auto_safe_lift"] = False
            rules = mem.get_symbolic_rules(spec.canonical_type)
            skills.configure_params(priors)

            rec = {
                "iteration": it,
                "transit_z": priors.get("transit_z"),
                "release_margin_z": priors.get("release_margin_z"),
                "n_rules": len(rules),
            }
            step_failed = False
            for cp in ont.decompose_checkpoints(spec):
                loc.clear_cache()
                objs, desc, _ = collect_scene(loc)
                plan = planner.plan(
                    task=cp, scene_description=desc, objects=objs,
                    extra_instructions=rules if rules else None,
                    kinematic_priors=priors,
                    canonical_type=spec.canonical_type,
                    task_spec=spec if len(ont.decompose_checkpoints(spec)) == 1 else None,
                )
                if not plan or not cap.execute(plan, scorer=scorer, enable_step_integrity=False):
                    diag = scorer._build_diagnostic(
                        cap.last_failure_code or "F_COLLISION",
                        cap.last_error or "Step execution failed",
                        None,
                    )
                    mem.record_failure(spec.canonical_type, diag)
                    skills.home()
                    step_failed = True
                    consecutive = 0
                    rec.update({"step_failed": True, "failure_code": diag.failure_code})
                    break
                skills.home()

            if not step_failed:
                loc.clear_cache()
                post_det = loc.locate_all_detailed()
                vlm = scorer.evaluate(cmd, perceived_details=post_det, prefer_multimodal=True, allow_pointcloud_fallback=False)
                gt = oracle.evaluate(spec)
                rec.update({
                    "step_failed": False,
                    "oracle_success": bool(gt["success"]),
                    "oracle_score": float(gt["score"]),
                    "vlm_success": bool(vlm.get("success")),
                    "vlm_score": float(vlm.get("score", 0.0)),
                    "vlm_error": bool(vlm.get("evaluation_error")),
                })
                if vlm.get("success") and vlm.get("score", 0.0) >= 0.8:
                    consecutive += 1
                    if consecutive >= 2:
                        mem.mark_calibrated(spec.canonical_type, cmd)
                        calibrated = True
                        iterations.append(rec)
                        break
                else:
                    consecutive = 0
                    if vlm.get("diagnostic") is not None:
                        mem.record_failure(spec.canonical_type, vlm["diagnostic"])
                        rec["failure_code"] = vlm["diagnostic"].failure_code
            iterations.append(rec)
            print(f"  [Calib {item['id']}] it={it} {rec}", flush=True)
    finally:
        env.close()

    final_priors = mem.get_kinematic_priors(spec.canonical_type)
    final_priors.pop("auto_safe_lift", None)
    return {
        "calibrated": calibrated,
        "iterations_run": len(iterations),
        "iterations": iterations,
        "final_priors": final_priors,
        "learned_rules": mem.get_symbolic_rules(spec.canonical_type),
        "usage": {
            "planner": {"model": planner.model, **_stats_of(planner.client)},
            "vlm": {"model": scorer.model, **_stats_of(scorer.client)},
        },
    }


def _compare(per_a: dict, per_b: dict, tasks: list[str]) -> dict:
    map_a = {t: per_a[t]["seed_outcomes"] for t in tasks}
    map_b = {t: per_b[t]["seed_outcomes"] for t in tasks}
    mc = mcnemar_exact([x for t in tasks for x in map_a[t]], [x for t in tasks for x in map_b[t]])
    boot = task_stratified_paired_bootstrap_ci(map_a, map_b, n_boot=5000, seed=42)
    return {
        "n_matched": mc["n_matched"],
        "success_a": boot["success_a"],
        "success_b": boot["success_b"],
        "rate_a_pct": boot["rate_a_pct"],
        "rate_b_pct": boot["rate_b_pct"],
        "diff_pp": boot["diff_pp"],
        "task_stratified_bootstrap_ci95_pp": boot["ci95_pp"],
        "b_only_a": mc["b_only_a"],
        "c_only_b": mc["c_only_b"],
        "mcnemar_p_value": mc["p_value"],
    }


def _load(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save(payload: dict):
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-calib-iters", type=int, default=5)
    ap.add_argument("--tasks", type=str, default="", help="comma-separated task ids (default: all six)")
    ap.add_argument("--fresh", action="store_true", help="ignore previous partial results")
    args = ap.parse_args()

    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise SystemExit("OPENAI_API_KEY missing in .env")

    wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
    payload = {} if args.fresh else _load(OUT_PATH)
    payload.setdefault("protocol", {
        "question": "Does feedback calibration recover success when the safe-Z lift is disabled and priors start at naive defaults?",
        "default_priors": dict(DEFAULT_KINEMATIC_PRIORS),
        "calibration_seeds": "5000 + 100 * t_idx + it",
        "evaluation_seeds": "3000 + 100 * t_idx + tr",
        "trials_per_task": args.trials,
        "max_calib_iters": args.max_calib_iters,
        "both_arms": {"use_multiview": True, "use_ontology": True, "use_checkpoints": True, "auto_safe_lift": False},
        "arms": {
            "default_no_lift": "default priors, no rules",
            "calibrated_no_lift": "priors and rules learned by the no-lift calibration loop",
        },
    })
    calib = payload.setdefault("calibration", {})
    results = payload.setdefault("results_by_arm", {a: {} for a in ARMS})

    t0 = time.time()
    for t_idx, item in enumerate(BENCHMARK_TASKS):
        tid = item["id"]
        if wanted and tid not in wanted:
            continue
        if all(tid in results.get(a, {}) for a in ARMS):
            print(f"[skip] {tid} already done", flush=True)
            continue

        if tid not in calib:
            print(f"[Calibrate] {tid}", flush=True)
            calib[tid] = calibrate_without_safe_lift(t_idx, item, args.max_calib_iters)
            _save(payload)
        c = calib[tid]

        seeds = [3000 + t_idx * 100 + tr for tr in range(args.trials)]
        common = dict(cmd=item["synonym_tr"], canonical_cmd=item["command_tr"], seeds=seeds, workers=args.workers,
                      use_multiview=True, use_ontology=True, use_checkpoints=True, auto_safe_lift=False)
        results.setdefault("default_no_lift", {})[tid] = run_tier_parallel(
            priors=dict(DEFAULT_KINEMATIC_PRIORS), rules=None, **common)
        _save(payload)
        results.setdefault("calibrated_no_lift", {})[tid] = run_tier_parallel(
            priors=dict(c["final_priors"]), rules=(c["learned_rules"] or None), **common)
        _save(payload)
        print(f"[Eval {tid}] default={results['default_no_lift'][tid]['success']}/{args.trials} "
              f"calibrated={results['calibrated_no_lift'][tid]['success']}/{args.trials} "
              f"(final transit_z={c['final_priors'].get('transit_z')}, release={c['final_priors'].get('release_margin_z')}, "
              f"rules={len(c['learned_rules'])}) elapsed={time.time() - t0:.0f}s", flush=True)

    done = [it["id"] for it in BENCHMARK_TASKS if all(it["id"] in results.get(a, {}) for a in ARMS)]
    if not done:
        return
    summary = {}
    for a in ARMS:
        s = summarize_tier_across_tasks({t: results[a][t] for t in done})
        s.pop("matched_outcomes", None)
        s["wilson_ci95_pct"] = list(wilson_ci(s["total_success"], s["total_trials"]))
        summary[a] = s
    payload["tasks_completed"] = done
    payload["summary"] = summary
    payload["comparison_calibrated_vs_default"] = _compare(results["calibrated_no_lift"], results["default_no_lift"], done)

    live = _load(os.path.join(RES_DIR, "live_ablation_benchmark.json"))
    k4 = live.get("results_by_method", {}).get("neuro_symbolic_roboteach", {})
    if all(t in k4 and len(k4[t]["seed_outcomes"]) == len(results["calibrated_no_lift"][t]["seed_outcomes"]) for t in done):
        payload["reference_k4_with_safe_lift"] = {
            "calibrated_no_lift_vs_k4": _compare(results["calibrated_no_lift"], k4, done),
            "default_no_lift_vs_k4": _compare(results["default_no_lift"], k4, done),
        }

    usage = {"planner_calls": 0, "planner_prompt_tokens": 0, "planner_completion_tokens": 0,
             "vlm_calls": 0, "vlm_prompt_tokens": 0, "vlm_completion_tokens": 0}
    for t in done:
        cu = calib[t]["usage"]
        usage["planner_calls"] += cu["planner"]["call_count"]
        usage["planner_prompt_tokens"] += cu["planner"]["prompt_tokens"]
        usage["planner_completion_tokens"] += cu["planner"]["completion_tokens"]
        usage["vlm_calls"] += cu["vlm"]["call_count"]
        usage["vlm_prompt_tokens"] += cu["vlm"]["prompt_tokens"]
        usage["vlm_completion_tokens"] += cu["vlm"]["completion_tokens"]
        for a in ARMS:
            lu = results[a][t]["llm_usage"]
            usage["planner_calls"] += lu["calls"]
            usage["planner_prompt_tokens"] += lu["prompt_tokens"]
            usage["planner_completion_tokens"] += lu["completion_tokens"]
    payload["api_usage"] = usage
    _save(payload)

    cmp_ = payload["comparison_calibrated_vs_default"]
    print(f"WROTE {OUT_PATH}")
    print(f"calibrated_no_lift {cmp_['success_a']}/{cmp_['n_matched']} vs default_no_lift {cmp_['success_b']}/{cmp_['n_matched']} "
          f"diff={cmp_['diff_pp']} pp CI={cmp_['task_stratified_bootstrap_ci95_pp']} p={cmp_['mcnemar_p_value']:.4g}")
    for t in done:
        print(f"  {t}: default {results['default_no_lift'][t]['success']} | calibrated {results['calibrated_no_lift'][t]['success']} "
              f"| calibrated={calib[t]['calibrated']} iters={calib[t]['iterations_run']} final_transit_z={calib[t]['final_priors'].get('transit_z')}")
    print(f"API usage: {usage}")


if __name__ == "__main__":
    main()
