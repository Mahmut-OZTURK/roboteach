# main.py — RoboTeach v2: Neuro-Symbolic Closed-Loop Robot Manipulation & Self-Calibration
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation.env import RoboTeachEnv
from simulation.camera import SimCamera
from perception.vision_locator import VisionLocator
from calibration.manager import CalibrationManager
from calibration.memory import FailureDiagnostic
from skills.primitives import RobotSkills
from planner.progprompt import ProgPrompt
from planner.cap import CaP
from evaluation.vlm_scorer import VLMScorer


MAX_CALIBRATION_RETRIES = 6


def _collect_scene_objects(vision_locator: VisionLocator) -> list[dict]:
    details = vision_locator.locate_all_detailed()
    return [
        {"name": name, "position": info["position"], "z_top": info.get("z_top", info["position"][2] + 0.025)}
        for name, info in details.items()
    ]


def calibrate(
    task: str,
    env: RoboTeachEnv,
    skills: RobotSkills,
    planner: ProgPrompt,
    cap: CaP,
    scorer: VLMScorer,
    calib_mgr: CalibrationManager,
    vision_locator: VisionLocator,
    max_retries: int = MAX_CALIBRATION_RETRIES,
) -> bool:
    spec = calib_mgr.parse_task(task)
    task_type = spec.canonical_type.lower()
    initial_state = spec.to_scene_config()
    checkpoints = calib_mgr.ontology.decompose_checkpoints(spec)

    print(f"\n{'='*60}\n  CALIBRATION MODE: '{task}' -> [{spec.canonical_type}]\n{'='*60}")

    for attempt in range(1, max_retries + 1):
        print(f"\n--- Calibration Attempt {attempt}/{max_retries} ---")
        env.reset()
        env.randomize(initial_state)

        priors = calib_mgr.get_priors(task_type)
        rules = calib_mgr.get_structured_rules(task_type)
        skills.configure_params(priors)

        all_passed = True
        for cp_idx, checkpoint in enumerate(checkpoints):
            vision_locator.clear_cache()
            scene_desc = scorer.analyze_scene()
            scene_objects = _collect_scene_objects(vision_locator)
            if not scene_objects:
                skills.home()
                all_passed = False
                break

            plan = planner.plan(
                task=checkpoint,
                scene_description=scene_desc,
                objects=scene_objects,
                extra_instructions=rules if rules else None,
                kinematic_priors=priors,
                canonical_type=spec.canonical_type,
                task_spec=spec if len(checkpoints) == 1 else None,
            )
            if not plan:
                all_passed = False
                break

            exec_ok = cap.execute(plan, scorer=scorer, enable_step_integrity=True)
            if not exec_ok:
                diag = FailureDiagnostic(
                    failure_code=cap.last_failure_code or "F_EXEC_SYNTAX",
                    summary=cap.last_error or f"Execution failed on '{checkpoint}'",
                    param_updates={"transit_z": 0.66} if cap.last_failure_code == "F_COLLISION" else {},
                    symbolic_rule=(
                        "Lift to transit_z >= 0.66m before horizontal moves and output only valid primitive calls."
                    ),
                )
                calib_mgr.record_structured_failure(task_type, diag)
                skills.home()
                all_passed = False
                break

            skills.home()
            is_last = cp_idx == len(checkpoints) - 1
            if is_last:
                result = scorer.evaluate(task)
                if result.get("evaluation_error") or not result["success"] or result["score"] < 0.8:
                    if result.get("diagnostic") is not None:
                        calib_mgr.record_structured_failure(task_type, result["diagnostic"])
                    all_passed = False
                    break

        if all_passed:
            final_rules = calib_mgr.get_structured_rules(task_type)
            calib_mgr.save_calibration(task_type, task, final_rules)
            print(f"\nCALIBRATED [{spec.canonical_type}] in {attempt} attempt(s).")
            return True

    return False


def execute(
    task: str,
    env: RoboTeachEnv,
    skills: RobotSkills,
    planner: ProgPrompt,
    cap: CaP,
    scorer: VLMScorer,
    calib_mgr: CalibrationManager,
    vision_locator: VisionLocator,
    max_attempts: int = 3,
) -> bool:
    spec = calib_mgr.parse_task(task)
    task_type = spec.canonical_type.lower()
    initial_state = spec.to_scene_config()
    checkpoints = calib_mgr.ontology.decompose_checkpoints(spec)

    priors = calib_mgr.get_priors(task_type)
    rules = calib_mgr.get_structured_rules(task_type)
    skills.configure_params(priors)

    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Execute Attempt {attempt}/{max_attempts} [{spec.canonical_type}] ---")
        env.reset()
        env.randomize(initial_state)

        all_passed = True
        for cp_idx, checkpoint in enumerate(checkpoints):
            vision_locator.clear_cache()
            scene_desc = scorer.analyze_scene()
            scene_objects = _collect_scene_objects(vision_locator)
            if not scene_objects:
                all_passed = False
                break

            plan = planner.plan(
                task=checkpoint,
                scene_description=scene_desc,
                objects=scene_objects,
                extra_instructions=rules if rules else None,
                kinematic_priors=priors,
                canonical_type=spec.canonical_type,
                task_spec=spec if len(checkpoints) == 1 else None,
            )
            if not plan or not cap.execute(plan, scorer=scorer, enable_step_integrity=True):
                skills.home()
                all_passed = False
                break

            skills.home()
            if cp_idx == len(checkpoints) - 1:
                result = scorer.evaluate(task)
                if result.get("evaluation_error") or not result["success"] or result["score"] < 0.75:
                    all_passed = False
                    break

        if all_passed:
            return True
    return False


def main():
    headless = "--headless" in sys.argv
    env = RoboTeachEnv(headless=headless, include_container=True, include_obstacle=False)
    camera = SimCamera(headless=headless, physics_client_id=env.client)
    vision_locator = VisionLocator(camera, env.object_dimensions)
    calib_mgr = CalibrationManager()
    skills = RobotSkills(
        env.robot_id,
        env.objects,
        vision_locator,
        control_mode="pd",
        realtime=not headless,
        physics_client_id=env.client,
    )
    planner = ProgPrompt()
    cap = CaP(skills, physics_client_id=env.client)
    scorer = VLMScorer(camera)
    skills.home()

    while True:
        try:
            mode = input("\nMode (c=calibrate / e=execute / s=status / q=quit): ").strip().lower()
            if mode in ("quit", "q"):
                break
            if mode in ("status", "s"):
                for t in calib_mgr.list_calibrated():
                    print(f"  - {t}")
                continue
            if mode in ("calibrate", "c"):
                task = input("Task: ").strip()
                if task:
                    calibrate(task, env, skills, planner, cap, scorer, calib_mgr, vision_locator)
            elif mode in ("execute", "e"):
                task = input("Task: ").strip()
                if task:
                    execute(task, env, skills, planner, cap, scorer, calib_mgr, vision_locator)
        except KeyboardInterrupt:
            break
    skills.home()
    env.close()


if __name__ == "__main__":
    main()
