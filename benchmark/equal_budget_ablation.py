"""Equal-Budget Key-Matched Verbal Notes Injected Control (180 rollouts) + Persisted-Memory Audit."""
import argparse
import json
import os
import sys
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"), override=True)

from benchmark.ablation_suite import (
    BENCHMARK_TASKS,
    collect_scene,
    run_tier_parallel,
    summarize_tier_across_tasks,
)
from calibration.memory import DEFAULT_KINEMATIC_PRIORS
from calibration.ontology import CanonicalTaskOntology
from evaluation.stats import mcnemar_exact, task_stratified_paired_bootstrap_ci, wilson_ci
from evaluation.vlm_scorer import VLMScorer
from perception.vision_locator import VisionLocator
from planner.cap import CaP
from planner.openai_client import OpenAIChatClient
from planner.progprompt import ProgPrompt
from simulation.camera import SimCamera
from simulation.env import RoboTeachEnv

# 360-rollout persisted-memory comparison (seeds 90000+100*j+i); kept in-repo for reproducibility.
CODEX_EXTRACTED = os.path.join(ROOT, "benchmark", "results", "persisted_memory_360.json")


def _usd(prompt_tokens: int, completion_tokens: int) -> float:
    return round(prompt_tokens * 0.10e-6 + completion_tokens * 0.50e-6, 6)


def collect_verbal_tips(trials_cap: int = 12) -> tuple[dict[str, dict], dict]:
    """Collects non-empty verbal feedback notes on command_tr across calibration seeds 4000+100*j+it."""
    env = RoboTeachEnv(headless=True, seed=500, include_container=True, include_obstacle=False)
    camera = SimCamera(headless=True, physics_client_id=env.client)
    locator = VisionLocator(camera, env.object_dimensions)
    skills = RobotSkills_factory(env, locator)
    ont = CanonicalTaskOntology(llm_client=None)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    plan_client = OpenAIChatClient(api_key=api_key)
    vlm_client = OpenAIChatClient(api_key=api_key)
    planner = ProgPrompt(client=plan_client, model=model)
    cap = CaP(skills, physics_client_id=env.client)
    scorer = VLMScorer(camera, client=vlm_client, model=model, min_groq_interval_s=0.0)

    collected: dict[str, dict] = {}
    try:
        for t_idx, item in enumerate(BENCHMARK_TASKS):
            tid, cmd = item["id"], item["command_tr"]
            scene_cfg = ont.parse(cmd).to_scene_config()
            tips: list[str] = []
            iters_run = 0
            for it in range(1, trials_cap + 1):
                iters_run = it
                env.reset()
                env.randomize(scene_cfg, seed=4000 + t_idx * 100 + it)
                skills.home()
                skills.configure_params({
                    "transit_z": DEFAULT_KINEMATIC_PRIORS["transit_z"],
                    "release_margin_z": DEFAULT_KINEMATIC_PRIORS["release_margin_z"],
                    "auto_safe_lift": False,
                })
                locator.clear_cache()
                objs, desc, _ = collect_scene(locator)
                plan = planner.plan(
                    task=cmd, scene_description=desc, objects=objs,
                    extra_instructions=tips[-3:] if tips else None,
                    kinematic_priors=DEFAULT_KINEMATIC_PRIORS,
                )
                if plan:
                    cap.execute(plan, scorer=None, enable_step_integrity=False)
                skills.home()
                locator.clear_cache()
                post_det = locator.locate_all_detailed()
                vlm = scorer.evaluate(cmd, perceived_details=post_det, prefer_multimodal=True, allow_pointcloud_fallback=False)
                if not vlm.get("evaluation_error") and not (vlm.get("success") and vlm.get("score", 0.0) >= 0.8):
                    inst = (vlm.get("instruction") or "").strip()
                    if inst and inst not in tips:
                        tips.append(inst)
                if len(tips) >= 2 or (it >= 5 and len(tips) >= 1):
                    break
            collected[tid] = {
                "command_tr": cmd,
                "calib_iters_run": iters_run,
                "collected_tips": tips[-3:],
                "tip_count": len(tips[-3:]),
            }
            print(f"  [CalibTips] {tid}: iters={iters_run}, tips={len(tips[-3:])}", flush=True)
    finally:
        env.close()

    ps, vs = plan_client.get_stats(), vlm_client.get_stats()
    cal_usage = {
        "model": model,
        "planner_calls": ps["call_count"],
        "planner_prompt_tokens": ps["prompt_tokens"],
        "planner_completion_tokens": ps["completion_tokens"],
        "vlm_calls": vs["call_count"],
        "vlm_prompt_tokens": vs["prompt_tokens"],
        "vlm_completion_tokens": vs["completion_tokens"],
        "total_calls": ps["call_count"] + vs["call_count"],
        "total_prompt_tokens": ps["prompt_tokens"] + vs["prompt_tokens"],
        "total_completion_tokens": ps["completion_tokens"] + vs["completion_tokens"],
        "cost_usd": _usd(ps["prompt_tokens"] + vs["prompt_tokens"], ps["completion_tokens"] + vs["completion_tokens"]),
    }
    return collected, cal_usage


def RobotSkills_factory(env, locator):
    from skills.primitives import RobotSkills
    return RobotSkills(env.robot_id, env.objects, locator, control_mode="pd", realtime=False, physics_client_id=env.client)


def _compare_pair(per_task_a: dict, per_task_b: dict, tasks: list[str]) -> dict:
    map_a = {t: per_task_a[t]["seed_outcomes"] for t in tasks}
    map_b = {t: per_task_b[t]["seed_outcomes"] for t in tasks}
    flat_a = [x for t in tasks for x in map_a[t]]
    flat_b = [x for t in tasks for x in map_b[t]]
    mc = mcnemar_exact(flat_a, flat_b)
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


def _load_persisted_memory_from_codex() -> dict:
    if not os.path.exists(CODEX_EXTRACTED):
        return {}
    with open(CODEX_EXTRACTED, "r", encoding="utf-8") as fh:
        codex = json.load(fh)
    follow = codex.get("robotic_followup_summary", {})
    by_task = codex.get("memory_comparison_by_task", {})
    if by_task:
        t_names = [item["id"] for item in BENCHMARK_TASKS]
        mem_map = {t: by_task[t]["memory"]["seed_outcomes"] for t in t_names}
        nomem_map = {t: by_task[t]["no_memory"]["seed_outcomes"] for t in t_names}
        flat_m = [x for t in t_names for x in mem_map[t]]
        flat_n = [x for t in t_names for x in nomem_map[t]]
        mc_all = mcnemar_exact(flat_m, flat_n)
        boot_all = task_stratified_paired_bootstrap_ci(mem_map, nomem_map, n_boot=5000, seed=42)
        upd = ["T3_DIAGONAL_PAIR", "T5_UNSTACK_SWAP"]
        mc_upd = mcnemar_exact([x for t in upd for x in mem_map[t]], [x for t in upd for x in nomem_map[t]])
        boot_upd = task_stratified_paired_bootstrap_ci({t: mem_map[t] for t in upd}, {t: nomem_map[t] for t in upd}, n_boot=5000, seed=42)
    else:
        mc_all, boot_all, mc_upd, boot_upd = {}, {}, {}, {}
    return {
        "source": "Codex 360-rollout equal-budget persisted-memory experiment (seeds 90000+100*j+i)",
        "codex_reported_summary": {
            "memory_overall": follow.get("memory_overall"),
            "memory_updated_families": follow.get("memory_updated_families"),
            "memory_by_task": follow.get("memory_by_task"),
            "protocol": follow.get("protocol", {}).get("memory"),
            "api_usage": follow.get("api_usage", {}).get("memory"),
        },
        "recomputed_with_repo_stats": {
            "overall_n180": {"mcnemar": mc_all, "task_stratified_boot_5000": boot_all},
            "updated_t3_t5_n60": {"mcnemar": mc_upd, "task_stratified_boot_5000": boot_upd},
        },
        "per_task_counts": {
            t: {
                "memory_success": by_task[t]["memory"]["success"],
                "no_memory_success": by_task[t]["no_memory"]["success"],
                "trials": by_task[t]["memory"]["n"],
            }
            for t in by_task
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-check", action="store_true")
    args = parser.parse_args()

    res_dir = os.path.join(os.path.dirname(__file__), "results")
    with open(os.path.join(res_dir, "live_ablation_benchmark.json"), "r", encoding="utf-8") as fh:
        live = json.load(fh)
    fair_cap = live["results_by_method"]["fair_cap_multiview"]
    unkeyed_verbal = live["results_by_method"]["unstructured_verbal_reflexion"]
    full_ns = live["results_by_method"]["neuro_symbolic_roboteach"]
    persisted_mem = _load_persisted_memory_from_codex()

    if args.dry_check:
        print("DRY_CHECK_OK", "fair_cap_total=", sum(fair_cap[t]["success"] for t in fair_cap), "codex_mem=", bool(persisted_mem))
        return

    calib_tips, cal_usage = collect_verbal_tips()
    for tid, info in calib_tips.items():
        if not info["collected_tips"]:
            raise RuntimeError(f"Empty verbal_tips collected for {tid}")

    eval_results = {}
    eval_calls = eval_ptok = eval_ctok = 0
    for t_idx, item in enumerate(BENCHMARK_TASKS):
        tid, cmd = item["id"], item["command_tr"]
        seeds = [3000 + t_idx * 100 + tr for tr in range(args.trials)]
        tips = calib_tips[tid]["collected_tips"]
        res = run_tier_parallel(
            cmd=cmd, canonical_cmd=cmd, seeds=seeds, workers=args.workers,
            use_multiview=True, use_ontology=False, use_checkpoints=False,
            priors=dict(DEFAULT_KINEMATIC_PRIORS), auto_safe_lift=False, rules=tips,
        )
        res["injected_verbal_tips"] = tips
        res["calib_iters"] = calib_tips[tid]["calib_iters_run"]
        eval_results[tid] = res
        u = res["llm_usage"]
        eval_calls += u["calls"]
        eval_ptok += u["prompt_tokens"]
        eval_ctok += u["completion_tokens"]
        print(f"  [KeyMatchedVerbal {tid}] {res['success']}/{args.trials} vs FairCaP {fair_cap[tid]['success']}/{args.trials}", flush=True)

    summary = summarize_tier_across_tasks(eval_results)
    summary.pop("matched_outcomes", None)
    summary["wilson_ci95_pct"] = list(wilson_ci(summary["total_success"], summary["total_trials"]))

    all_tasks = [item["id"] for item in BENCHMARK_TASKS]
    t4_t6 = all_tasks[3:]
    eval_usage = {
        "model": os.getenv("OPENAI_MODEL", "gpt-6-luna"),
        "calls": eval_calls,
        "prompt_tokens": eval_ptok,
        "completion_tokens": eval_ctok,
        "cost_usd": _usd(eval_ptok, eval_ctok),
    }
    total_usage = {
        "model": eval_usage["model"],
        "total_calls": cal_usage["total_calls"] + eval_calls,
        "total_prompt_tokens": cal_usage["total_prompt_tokens"] + eval_ptok,
        "total_completion_tokens": cal_usage["total_completion_tokens"] + eval_ctok,
        "total_cost_usd": round(cal_usage["cost_usd"] + eval_usage["cost_usd"], 6),
    }

    payload = {
        "protocol": {
            "arm_name": "equal_budget_key_matched_verbal",
            "command_source": "canonical command_tr (identical to fair_cap_multiview)",
            "seeds_formula": "3000 + 100 * t_idx + tr (30 per task, 180 total)",
            "use_multiview": True,
            "use_ontology": False,
            "use_checkpoints": False,
            "auto_safe_lift": False,
            "priors": dict(DEFAULT_KINEMATIC_PRIORS),
            "injected_rules": "non-empty verbal_tips[-3:] collected on command_tr",
        },
        "calibration_verbal_tips": calib_tips,
        "results_by_task": eval_results,
        "summary": summary,
        "comparisons_n180": {
            "key_matched_verbal_vs_fair_cap": _compare_pair(eval_results, fair_cap, all_tasks),
            "key_matched_verbal_vs_unkeyed_verbal_k3": _compare_pair(eval_results, unkeyed_verbal, all_tasks),
            "full_ns_k4_vs_key_matched_verbal": _compare_pair(full_ns, eval_results, all_tasks),
        },
        "comparisons_t4_t6_n90": {
            "key_matched_verbal_vs_fair_cap": _compare_pair(eval_results, fair_cap, t4_t6),
            "full_ns_k4_vs_key_matched_verbal": _compare_pair(full_ns, eval_results, t4_t6),
        },
        "api_usage": {
            "calibration": cal_usage,
            "evaluation_180": eval_usage,
            "total_new_api_spend": total_usage,
        },
        "equal_budget_persisted_memory_360": persisted_mem,
    }
    out_path = os.path.join(res_dir, "equal_budget_verbal_and_memory.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    cmp_fc = payload["comparisons_n180"]["key_matched_verbal_vs_fair_cap"]
    print(f"WROTE: {out_path}")
    print(f"KeyMatchedVerbal={cmp_fc['success_a']}/180 vs FairCaP={cmp_fc['success_b']}/180 diff={cmp_fc['diff_pp']} pp CI={cmp_fc['task_stratified_bootstrap_ci95_pp']} p={cmp_fc['mcnemar_p_value']:.6g}")
    print(f"Total API spend: ${total_usage['total_cost_usd']:.6f} ({total_usage['total_prompt_tokens']} in, {total_usage['total_completion_tokens']} out)")


if __name__ == "__main__":
    main()
