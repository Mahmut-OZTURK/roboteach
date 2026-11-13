# runner.py — Reproducible Scientific Benchmark Suite for RoboTeach v2 (n=30, Cross-Family VLM & DR)
import argparse
import json
import math
import os
import sys
import time
import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.env import RoboTeachEnv
from simulation.camera import SimCamera
from perception.vision_locator import VisionLocator
from skills.primitives import RobotSkills
from calibration.ontology import CanonicalTaskOntology
from calibration.memory import CalibrationMemory, DEFAULT_KINEMATIC_PRIORS
from evaluation.oracle import GroundTruthOracle, compute_judge_agreement
from evaluation.stats import mcnemar_exact, percentile_summary
from evaluation.vlm_scorer import VLMScorer
from planner.openai_client import OpenAIChatClient
from planner.progprompt import ProgPrompt
from planner.cap import CaP
from benchmark.ablation_suite import (
    BENCHMARK_TASKS,
    SingleViewLocatorAdapter,
    collect_scene,
    describe_perceived_scene,
    freeform_llm_task_key,
    naive_single_view_locate,
    run_component_and_dr_ablation,
    run_tier_parallel,
    summarize_tier_across_tasks,
)


def run_perception_benchmark(num_scenes: int = 60, perturbed: bool = False) -> dict:
    """Evaluates 3D localization RMSE (mm) and detection recall across scattered and occluded scenes."""
    env = RoboTeachEnv(headless=True, seed=100, include_container=True, include_obstacle=True)
    camera = SimCamera(headless=True, physics_client_id=env.client)
    locator = VisionLocator(camera, env.object_dimensions)
    skills = RobotSkills(env.robot_id, env.objects, locator, control_mode="pd", realtime=False, physics_client_id=env.client)
    skills.home()

    naive_errs_unocc, prop_errs_unocc = [], []
    naive_hits_occ, prop_hits_occ = 0, 0
    occ_total = 0
    prop_errs_occ = []
    target_names = ["red_cube", "blue_cube", "green_cube", "yellow_cube", "brown_box", "obstacle_wall"]

    try:
        for i in range(num_scenes):
            camera.set_perturbation(enabled=perturbed, seed=9000 + i, depth_noise_mm=3.0, extrinsics_mm=3.0, extrinsics_deg=1.5)
            env.reset()
            if i % 2 == 0:
                env.randomize({"stacks": [], "has_obstacle": True}, seed=1000 + i, domain_randomization=perturbed)
                locator.clear_cache()
                top_frame = camera.capture_rgbd("top")
                prop_all = locator.locate_all_detailed()
                for name in target_names:
                    gt_pos, _ = p.getBasePositionAndOrientation(env.objects[name], physicsClientId=env.client)
                    est_naive = naive_single_view_locate(camera, locator, name, frame=top_frame)
                    est_prop = prop_all.get(name, {}).get("position")
                    if est_naive is not None:
                        naive_errs_unocc.append(math.dist(est_naive, gt_pos) * 1000.0)
                    if est_prop is not None:
                        prop_errs_unocc.append(math.dist(est_prop, gt_pos) * 1000.0)
            else:
                env.randomize({"stacks": [["red_cube", "blue_cube"], ["green_cube", "red_cube"]], "has_obstacle": False}, seed=2000 + i)
                locator.clear_cache()
                top_frame = camera.capture_rgbd("top")
                prop_all = locator.locate_all_detailed()
                for occ_cube in ("blue_cube", "red_cube"):
                    occ_total += 1
                    gt_pos, _ = p.getBasePositionAndOrientation(env.objects[occ_cube], physicsClientId=env.client)
                    est_naive = naive_single_view_locate(camera, locator, occ_cube, frame=top_frame)
                    est_prop = prop_all.get(occ_cube, {}).get("position")
                    if est_naive is not None and math.dist(est_naive, gt_pos) < 0.04:
                        naive_hits_occ += 1
                    if est_prop is not None and math.dist(est_prop, gt_pos) < 0.04:
                        prop_hits_occ += 1
                        prop_errs_occ.append(math.dist(est_prop, gt_pos) * 1000.0)
    finally:
        env.close()

    return {
        "num_scenes": num_scenes,
        "perturbed": perturbed,
        "unoccluded_rmse_mm_naive": round(float(np.sqrt(np.mean(np.square(naive_errs_unocc)))), 2),
        "unoccluded_rmse_mm_proposed": round(float(np.sqrt(np.mean(np.square(prop_errs_unocc)))), 2),
        "unoccluded_mae_mm_naive": round(float(np.mean(naive_errs_unocc)), 2),
        "unoccluded_mae_mm_proposed": round(float(np.mean(prop_errs_unocc)), 2),
        "occluded_recall_naive_pct": round(100.0 * naive_hits_occ / max(1, occ_total), 1),
        "occluded_recall_proposed_pct": round(100.0 * prop_hits_occ / max(1, occ_total), 1),
        "occluded_rmse_mm_proposed": round(float(np.sqrt(np.mean(np.square(prop_errs_occ)))), 2) if prop_errs_occ else None,
        "occluded_queries": occ_total,
    }


def _calibrate_task(
    t_idx: int,
    item: dict,
    env: RoboTeachEnv,
    mv_locator: VisionLocator,
    sv_locator: SingleViewLocatorAdapter,
    skills_sv: RobotSkills,
    skills_mv: RobotSkills,
    cap_sv: CaP,
    cap_mv: CaP,
    ont: CanonicalTaskOntology,
    oracle: GroundTruthOracle,
    planner: ProgPrompt,
    scorer: VLMScorer,
    max_calib_iters: int,
    calib_judge_logs: dict,
) -> dict:
    """Runs K2 verbal calibration and K4 neuro-symbolic calibration using 100% multimodal Qwen VLM critic."""
    tid, cmd, syn_cmd = item["id"], item["command_tr"], item["synonym_tr"]
    spec, syn_spec = ont.parse(cmd), ont.parse(syn_cmd)
    scene_cfg = spec.to_scene_config()

    calib_key = freeform_llm_task_key(planner.client, cmd, model=planner.model)
    test_key = freeform_llm_task_key(planner.client, syn_cmd, model=planner.model)
    verbal_tips: list[str] = []
    uv_calib_attempts = 0
    for it in range(1, max_calib_iters + 1):
        uv_calib_attempts = it
        env.reset()
        env.randomize(scene_cfg, seed=4000 + t_idx * 100 + it)
        skills_mv.home()
        skills_mv.configure_params({
            "transit_z": DEFAULT_KINEMATIC_PRIORS["transit_z"],
            "release_margin_z": DEFAULT_KINEMATIC_PRIORS["release_margin_z"],
            "auto_safe_lift": False,
        })
        mv_locator.clear_cache()
        objs, desc, _ = collect_scene(mv_locator)
        plan = planner.plan(
            task=cmd,
            scene_description=desc,
            objects=objs,
            extra_instructions=verbal_tips[-3:] if verbal_tips else None,
            kinematic_priors=DEFAULT_KINEMATIC_PRIORS,
        )
        if plan:
            cap_mv.execute(plan, scorer=None, enable_step_integrity=False)
        skills_mv.home()

        mv_locator.clear_cache()
        post_det = mv_locator.locate_all_detailed()
        vlm = scorer.evaluate(cmd, perceived_details=post_det, prefer_multimodal=True, allow_pointcloud_fallback=False)
        gt = oracle.evaluate(spec)
        if not vlm.get("evaluation_error"):
            calib_judge_logs["o_lbl"].append(bool(gt["success"]))
            calib_judge_logs["v_lbl"].append(bool(vlm["success"]))
            calib_judge_logs["o_scr"].append(float(gt["score"]))
            calib_judge_logs["v_scr"].append(float(vlm["score"]))
        if vlm.get("success") and vlm.get("score", 0.0) >= 0.8:
            break
        if vlm.get("instruction"):
            verbal_tips.append(vlm["instruction"])

    tmp_mem_dir = os.path.join(os.path.dirname(__file__), "results", "calib_state")
    mem = CalibrationMemory(storage_dir=tmp_mem_dir)
    entry_path = mem._path(spec.canonical_type)
    if os.path.exists(entry_path):
        os.remove(entry_path)

    ns_calib_iters = 0
    ns_calibrated = False
    consecutive_passes = 0
    iter_scores = []
    best_priors = None
    best_rules = None
    for it in range(1, max_calib_iters + 1):
        ns_calib_iters = it
        env.reset()
        env.randomize(scene_cfg, seed=5000 + t_idx * 100 + it)
        skills_mv.home()
        priors = mem.get_kinematic_priors(spec.canonical_type)
        priors["auto_safe_lift"] = True
        rules = mem.get_symbolic_rules(spec.canonical_type)
        skills_mv.configure_params(priors)

        checkpoints = ont.decompose_checkpoints(spec)
        step_failed = False
        for cp in checkpoints:
            mv_locator.clear_cache()
            objs, desc, _ = collect_scene(mv_locator)
            plan = planner.plan(
                task=cp,
                scene_description=desc,
                objects=objs,
                extra_instructions=rules if rules else None,
                kinematic_priors=priors,
                canonical_type=spec.canonical_type,
                task_spec=spec if len(checkpoints) == 1 else None,
            )
            if not plan or not cap_mv.execute(plan, scorer=scorer, enable_step_integrity=False):
                diag = scorer._build_diagnostic(
                    cap_mv.last_failure_code or "F_COLLISION",
                    cap_mv.last_error or "Step execution failed",
                    None,
                )
                mem.record_failure(spec.canonical_type, diag)
                skills_mv.home()
                step_failed = True
                consecutive_passes = 0
                iter_scores.append(0.0)
                break
            skills_mv.home()

        if not step_failed:
            mv_locator.clear_cache()
            post_det = mv_locator.locate_all_detailed()
            vlm = scorer.evaluate(cmd, perceived_details=post_det, prefer_multimodal=True, allow_pointcloud_fallback=False)
            gt = oracle.evaluate(spec)
            if not vlm.get("evaluation_error"):
                calib_judge_logs["o_lbl"].append(bool(gt["success"]))
                calib_judge_logs["v_lbl"].append(bool(vlm["success"]))
                calib_judge_logs["o_scr"].append(float(gt["score"]))
                calib_judge_logs["v_scr"].append(float(vlm["score"]))
            iter_scores.append(float(gt["score"]))
            if vlm.get("success") and vlm.get("score", 0.0) >= 0.8:
                best_priors = dict(priors)
                best_rules = list(rules) if rules else None
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    mem.mark_calibrated(spec.canonical_type, cmd)
                    ns_calibrated = True
                    break
            else:
                consecutive_passes = 0
                if vlm.get("diagnostic") is not None:
                    mem.record_failure(spec.canonical_type, vlm["diagnostic"])

    final_priors = mem.get_kinematic_priors(spec.canonical_type)
    final_rules = mem.get_symbolic_rules(spec.canonical_type)
    if not ns_calibrated and best_priors is not None and not final_rules:
        final_priors = best_priors

    return {
        "synonym_pair": {
            "task_id": tid,
            "freeform_calib_key": calib_key,
            "freeform_synonym_key": test_key,
            "freeform_hit": calib_key == test_key,
            "ontology_canonical_type": spec.canonical_type,
            "ontology_hit": spec.canonical_type == syn_spec.canonical_type,
        },
        "k2_calib_iters": uv_calib_attempts,
        "k2_retrieved_tips": (verbal_tips[-3:] if calib_key == test_key and verbal_tips else None),
        "k4_curve": {
            "calibrated": ns_calibrated,
            "iterations_to_converge": ns_calib_iters,
            "iteration_gt_scores": iter_scores,
            "final_priors": final_priors,
            "learned_rules": final_rules,
        },
    }


def _build_main_payload(trials_per_task, results_by_method, calibration_curves, synonym_routing_log, calib_judge_logs, synthesizer_model, judge_model="gpt-4.1") -> dict:
    summary = {}
    matched = {}
    for m, per_task in results_by_method.items():
        s = summarize_tier_across_tasks(per_task)
        matched[m] = s.pop("matched_outcomes")
        summary[m] = s

    ns_m = matched.get("neuro_symbolic_roboteach", [])
    mcnemar = {}
    for cmp_key in ("zero_shot_open_loop", "fair_cap_multiview", "unstructured_verbal_reflexion"):
        other = matched.get(cmp_key, [])
        if ns_m and len(other) == len(ns_m):
            mcnemar[f"ns_vs_{cmp_key}"] = mcnemar_exact(ns_m, other)

    freeform_hits = sum(1 for r in synonym_routing_log if r["freeform_hit"])
    ont_hits = sum(1 for r in synonym_routing_log if r["ontology_hit"])
    calib_agr = compute_judge_agreement(
        calib_judge_logs["o_lbl"], calib_judge_logs["v_lbl"],
        calib_judge_logs["o_scr"], calib_judge_logs["v_scr"],
    )
    return {
        "trials_per_task": trials_per_task,
        "total_tasks": len(BENCHMARK_TASKS),
        "models": {
            "policy_synthesizer": synthesizer_model,
            "cross_family_multimodal_vlm_judge": judge_model,
        },
        "results_by_method": results_by_method,
        "summary": summary,
        "paired_mcnemar_tests": mcnemar,
        "synonym_routing": {
            "total_pairs": len(synonym_routing_log),
            "freeform_hit_pct": round(100.0 * freeform_hits / max(1, len(synonym_routing_log)), 1),
            "ontology_hit_pct": round(100.0 * ont_hits / max(1, len(synonym_routing_log)), 1),
            "pairs": synonym_routing_log,
        },
        "calibration_curves": calibration_curves,
        "calibration_vlm_vs_oracle_agreement": calib_agr,
    }


def run_live_ablation_and_vlm_validation(
    trials_per_task: int = 30,
    max_calib_iters: int = 5,
    out_path: str | None = None,
    task_filter: set[str] | None = None,
    workers: int = 4,
    preserve_baselines: bool = False,
) -> dict:
    env = RoboTeachEnv(headless=True, seed=500, include_container=True, include_obstacle=False)
    camera = SimCamera(headless=True, physics_client_id=env.client)
    mv_locator = VisionLocator(camera, env.object_dimensions)
    sv_locator = SingleViewLocatorAdapter(camera, mv_locator)
    skills_sv = RobotSkills(env.robot_id, env.objects, sv_locator, control_mode="pd", realtime=False, physics_client_id=env.client)
    skills_mv = RobotSkills(env.robot_id, env.objects, mv_locator, control_mode="pd", realtime=False, physics_client_id=env.client)
    ont = CanonicalTaskOntology(llm_client=None)
    oracle = GroundTruthOracle(env)
    planner = ProgPrompt()
    cap_sv = CaP(skills_sv, physics_client_id=env.client)
    cap_mv = CaP(skills_mv, physics_client_id=env.client)
    scorer = VLMScorer(camera)

    results_by_method = {
        "zero_shot_open_loop": {},
        "fair_cap_multiview": {},
        "unstructured_verbal_reflexion": {},
        "neuro_symbolic_roboteach": {},
    }
    calibration_curves = {}
    synonym_routing_log = []
    calib_judge_logs = {"o_lbl": [], "v_lbl": [], "o_scr": [], "v_scr": []}

    if (task_filter or preserve_baselines) and out_path and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as fh:
            prev = json.load(fh)
        for m in results_by_method:
            results_by_method[m].update(prev.get("results_by_method", {}).get(m, {}))
        if task_filter:
            calibration_curves.update(prev.get("calibration_curves", {}))
            synonym_routing_log = [p for p in prev.get("synonym_routing", {}).get("pairs", []) if p.get("task_id") not in task_filter]

    calib_per_task = {}
    try:
        for t_idx, item in enumerate(BENCHMARK_TASKS):
            tid = item["id"]
            if task_filter and tid not in task_filter:
                continue
            print(f"\n=== Calibrating {tid} ({planner.model} + Cross-Family {scorer.model}) ===", flush=True)
            c_res = _calibrate_task(
                t_idx, item, env, mv_locator, sv_locator, skills_sv, skills_mv,
                cap_sv, cap_mv, ont, oracle, planner, scorer, max_calib_iters, calib_judge_logs,
            )
            calib_per_task[tid] = c_res
            synonym_routing_log.append(c_res["synonym_pair"])
            calibration_curves[tid] = c_res["k4_curve"]
    finally:
        env.close()

    for t_idx, item in enumerate(BENCHMARK_TASKS):
        tid, cmd, syn_cmd = item["id"], item["command_tr"], item["synonym_tr"]
        if task_filter and tid not in task_filter:
            continue
        c_res = calib_per_task[tid]
        seeds = [3000 + t_idx * 100 + tr for tr in range(trials_per_task)]

        has_prev_baselines = (
            preserve_baselines
            and tid in results_by_method["zero_shot_open_loop"]
            and tid in results_by_method["fair_cap_multiview"]
            and tid in results_by_method["unstructured_verbal_reflexion"]
        )
        if has_prev_baselines:
            r2 = results_by_method["unstructured_verbal_reflexion"][tid]
        else:
            results_by_method["zero_shot_open_loop"][tid] = run_tier_parallel(
                cmd=cmd, canonical_cmd=cmd, seeds=seeds, workers=workers,
                use_multiview=False, use_ontology=False, use_checkpoints=False,
                priors=dict(DEFAULT_KINEMATIC_PRIORS), auto_safe_lift=False, rules=None,
            )
            results_by_method["fair_cap_multiview"][tid] = run_tier_parallel(
                cmd=cmd, canonical_cmd=cmd, seeds=seeds, workers=workers,
                use_multiview=True, use_ontology=False, use_checkpoints=False,
                priors=dict(DEFAULT_KINEMATIC_PRIORS), auto_safe_lift=False, rules=None,
            )
            r2 = run_tier_parallel(
                cmd=syn_cmd, canonical_cmd=cmd, seeds=seeds, workers=workers,
                use_multiview=True, use_ontology=False, use_checkpoints=False,
                priors=dict(DEFAULT_KINEMATIC_PRIORS), auto_safe_lift=False, rules=c_res["k2_retrieved_tips"],
            )
            r2["calib_iters"] = c_res["k2_calib_iters"]
            r2["synonym_cache_hit"] = c_res["synonym_pair"]["freeform_hit"]
            results_by_method["unstructured_verbal_reflexion"][tid] = r2

        r4 = run_tier_parallel(
            cmd=syn_cmd, canonical_cmd=cmd, seeds=seeds, workers=workers,
            use_multiview=True, use_ontology=True, use_checkpoints=True,
            priors=dict(c_res["k4_curve"]["final_priors"]), auto_safe_lift=True,
            rules=c_res["k4_curve"]["learned_rules"] or None,
        )
        r4["calib_iters"] = c_res["k4_curve"]["iterations_to_converge"]
        results_by_method["neuro_symbolic_roboteach"][tid] = r4

        print(
            f"[{tid}] ZeroShot={results_by_method['zero_shot_open_loop'][tid]['success']}/{trials_per_task} | "
            f"FairCaP={results_by_method['fair_cap_multiview'][tid]['success']}/{trials_per_task} | "
            f"Verbal={r2['success']}/{trials_per_task} | FullNS={r4['success']}/{trials_per_task}",
            flush=True,
        )
        if out_path:
            payload = _build_main_payload(trials_per_task, results_by_method, calibration_curves, synonym_routing_log, calib_judge_logs, planner.model, scorer.model)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)

    return _build_main_payload(trials_per_task, results_by_method, calibration_curves, synonym_routing_log, calib_judge_logs, planner.model, scorer.model)


def run_judge_agreement_benchmark(episodes_per_task: int = 20, live_path: str | None = None, out_path: str | None = None) -> dict:
    """Evaluates Heterogeneous Dense VLM (gpt-4.1) and Homogeneous Reasoning Self-Critic (gpt-6-luna) vs Oracle across N=120 episodes."""
    env = RoboTeachEnv(headless=True, seed=700, include_container=True, include_obstacle=False)
    camera = SimCamera(headless=True, physics_client_id=env.client)
    mv_locator = VisionLocator(camera, env.object_dimensions)
    skills_mv = RobotSkills(env.robot_id, env.objects, mv_locator, control_mode="pd", realtime=False, physics_client_id=env.client)
    ont = CanonicalTaskOntology(llm_client=None)
    oracle = GroundTruthOracle(env)
    planner = ProgPrompt()
    cap_mv = CaP(skills_mv, physics_client_id=env.client)

    cross_scorer = VLMScorer(camera)
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    intra_client = OpenAIChatClient(api_key=openai_key) if openai_key else None
    intra_model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    intra_scorer = VLMScorer(camera, client=intra_client, model=intra_model, min_groq_interval_s=0.0) if intra_client else None

    curves = {}
    if live_path and os.path.exists(live_path):
        with open(live_path, "r", encoding="utf-8") as fh:
            curves = json.load(fh).get("calibration_curves", {})

    logs = {
        "cross": {"o_lbl": [], "v_lbl": [], "o_scr": [], "v_scr": [], "lat_ms": [], "errs": 0},
        "intra": {"o_lbl": [], "v_lbl": [], "o_scr": [], "v_scr": [], "lat_ms": [], "errs": 0},
    }

    try:
        for t_idx, item in enumerate(BENCHMARK_TASKS):
            tid, cmd = item["id"], item["command_tr"]
            spec = ont.parse(cmd)
            scene_cfg = spec.to_scene_config()
            c_info = curves.get(tid, {})
            learned_rules = c_info.get("learned_rules") or None
            final_priors = dict(c_info.get("final_priors") or DEFAULT_KINEMATIC_PRIORS)

            for ep in range(episodes_per_task):
                use_calibrated = ep >= (episodes_per_task // 2)
                env.reset()
                env.randomize(scene_cfg, seed=7000 + t_idx * 100 + ep)
                skills_mv.home()
                if use_calibrated:
                    p_cfg = dict(final_priors)
                    p_cfg["auto_safe_lift"] = True
                    skills_mv.configure_params(p_cfg)
                    checkpoints = ont.decompose_checkpoints(spec)
                    for cp in checkpoints:
                        mv_locator.clear_cache()
                        objs, desc, _ = collect_scene(mv_locator)
                        plan = planner.plan(
                            task=cp, scene_description=desc, objects=objs,
                            extra_instructions=learned_rules, kinematic_priors=p_cfg,
                            canonical_type=spec.canonical_type,
                            task_spec=spec if len(checkpoints) == 1 else None,
                        )
                        if not plan or not cap_mv.execute(plan, scorer=None, enable_step_integrity=False):
                            skills_mv.home()
                            break
                        skills_mv.home()
                else:
                    skills_mv.configure_params({"transit_z": 0.56, "release_margin_z": 0.075, "auto_safe_lift": False})
                    if ep % 2 == 1:
                        mv_locator.clear_cache()
                        objs, desc, _ = collect_scene(mv_locator)
                        plan = planner.plan(task=cmd, scene_description=desc, objects=objs, kinematic_priors=DEFAULT_KINEMATIC_PRIORS)
                        if plan:
                            cap_mv.execute(plan, scorer=None, enable_step_integrity=False)
                        skills_mv.home()

                gt = oracle.evaluate(spec)
                mv_locator.clear_cache()
                post_det = mv_locator.locate_all_detailed()

                t0 = time.perf_counter()
                vc = cross_scorer.evaluate(cmd, perceived_details=post_det, prefer_multimodal=True, allow_pointcloud_fallback=False)
                fallback_vc_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                logs["cross"]["lat_ms"].append(float(vc.get("api_latency_ms") or fallback_vc_ms))
                if vc.get("evaluation_error"):
                    logs["cross"]["errs"] += 1
                else:
                    logs["cross"]["o_lbl"].append(bool(gt["success"]))
                    logs["cross"]["v_lbl"].append(bool(vc["success"]))
                    logs["cross"]["o_scr"].append(float(gt["score"]))
                    logs["cross"]["v_scr"].append(float(vc["score"]))

                if intra_scorer is not None:
                    t1 = time.perf_counter()
                    vi = intra_scorer.evaluate(cmd, perceived_details=post_det, prefer_multimodal=True, allow_pointcloud_fallback=False)
                    fallback_vi_ms = round((time.perf_counter() - t1) * 1000.0, 2)
                    logs["intra"]["lat_ms"].append(float(vi.get("api_latency_ms") or fallback_vi_ms))
                    if vi.get("evaluation_error"):
                        logs["intra"]["errs"] += 1
                    else:
                        logs["intra"]["o_lbl"].append(bool(gt["success"]))
                        logs["intra"]["v_lbl"].append(bool(vi["success"]))
                        logs["intra"]["o_scr"].append(float(gt["score"]))
                        logs["intra"]["v_scr"].append(float(vi["score"]))
            print(f"  [JudgeAudit] {tid}: cross_n={len(logs['cross']['o_lbl'])}, intra_n={len(logs['intra']['o_lbl'])}", flush=True)
    finally:
        env.close()

    res = {
        "target_episodes": len(BENCHMARK_TASKS) * episodes_per_task,
        "cross_family_critic": {
            "model": cross_scorer.model,
            "errors": logs["cross"]["errs"],
            "agreement": compute_judge_agreement(logs["cross"]["o_lbl"], logs["cross"]["v_lbl"], logs["cross"]["o_scr"], logs["cross"]["v_scr"]),
            "latency_ms": percentile_summary(logs["cross"]["lat_ms"]),
        },
        "intra_family_critic": {
            "model": intra_model,
            "errors": logs["intra"]["errs"],
            "agreement": compute_judge_agreement(logs["intra"]["o_lbl"], logs["intra"]["v_lbl"], logs["intra"]["o_scr"], logs["intra"]["v_scr"]),
            "latency_ms": percentile_summary(logs["intra"]["lat_ms"]),
        },
    }
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["perception", "live", "component", "judge", "all"], default="all")
    parser.add_argument("--scenes", type=int, default=60)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--preserve-baselines", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    live_path = os.path.join(out_dir, "live_ablation_benchmark.json")
    comp_path = os.path.join(out_dir, "component_and_dr_ablation.json")
    judge_path = os.path.join(out_dir, "judge_audit_120.json")
    t_filter = {t.strip() for t in args.tasks.split(",") if t.strip()} if args.tasks else None

    if args.mode in ("perception", "all"):
        perc_nom = run_perception_benchmark(num_scenes=args.scenes, perturbed=False)
        perc_pert = run_perception_benchmark(num_scenes=args.scenes, perturbed=True)
        with open(os.path.join(out_dir, "perception_benchmark.json"), "w", encoding="utf-8") as fh:
            json.dump({"nominal": perc_nom, "perturbed": perc_pert}, fh, indent=2)
        print("Perception Nominal:", json.dumps(perc_nom, indent=2), flush=True)
        print("Perception Perturbed:", json.dumps(perc_pert, indent=2), flush=True)

    calib_info = None
    if args.mode in ("live", "all"):
        live = run_live_ablation_and_vlm_validation(
            trials_per_task=args.trials, out_path=live_path, task_filter=t_filter, workers=args.workers,
            preserve_baselines=args.preserve_baselines,
        )
        calib_info = live.get("calibration_curves")
        print("Live Summary:", json.dumps(live["summary"], indent=2), flush=True)

    if args.mode in ("component", "all"):
        if calib_info is None and os.path.exists(live_path):
            with open(live_path, "r", encoding="utf-8") as fh:
                calib_info = json.load(fh).get("calibration_curves")
        comp = run_component_and_dr_ablation(
            trials_per_task=args.trials, calib_info=calib_info, out_path=comp_path,
            task_filter=t_filter, workers=args.workers,
        )
        print("Component+DR Summary:", json.dumps(comp["summary"], indent=2), flush=True)

    if args.mode in ("judge", "all"):
        res = run_judge_agreement_benchmark(episodes_per_task=args.episodes, live_path=live_path, out_path=judge_path)
        print("Judge Audit:", json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
