# progprompt.py — Neuro-Symbolic Code-as-Policies Synthesizer Conditioned on Structured Memory
import os
import time
from groq import Groq
from dotenv import load_dotenv

from calibration.memory import DEFAULT_KINEMATIC_PRIORS
from planner.openai_client import OpenAIChatClient

load_dotenv(override=True)


class ProgPrompt:
    """
    Synthesizes flat, AST-safe Python manipulation policies using the primitive skill API.
    Unlike hardcoded demonstration templates, geometric clearance, release margins, and
    causal invariants are injected dynamically from CalibrationMemory.
    """

    def __init__(self, client=None, model: str | None = None):
        openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if client is not None:
            self.client = client
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-6-luna")
        elif openai_key:
            self.client = OpenAIChatClient(api_key=openai_key)
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-6-luna")
        elif groq_key:
            self.client = Groq(api_key=groq_key, max_retries=0, timeout=25.0)
            self.model = model or "openai/gpt-oss-120b"
        else:
            self.client = None
            self.model = model or "openai/gpt-oss-120b"

    def plan(
        self,
        task: str,
        scene_description: str,
        objects: list[dict],
        robot_position: list[float] | None = None,
        extra_instructions: list[str] | None = None,
        kinematic_priors: dict | None = None,
        canonical_type: str | None = None,
        task_spec=None,
    ) -> str | None:
        priors = dict(DEFAULT_KINEMATIC_PRIORS)
        if kinematic_priors:
            priors.update(kinematic_priors)

        objects_str = "\n".join(
            f"  - {obj['name']}: center={[round(x, 3) for x in obj['position']]}"
            + (f", z_top={round(obj['z_top'], 3)}" if "z_top" in obj else "")
            for obj in objects
        )
        obj_names = [obj["name"] for obj in objects]

        rules_block = ""
        if extra_instructions:
            rules_block = "\nCALIBRATED CAUSAL CONSTRAINTS (MUST OBEY):\n" + "\n".join(
                f"- {r}" for r in extra_instructions
            )

        spec_block = ""
        if task_spec is not None:
            fam = task_spec.canonical_type
            if fam in ("STACK_ORDERED", "OBSTACLE_CLEARANCE") and len(task_spec.ordered_objects) >= 2:
                base = task_spec.ordered_objects[0]
                uppers = task_spec.ordered_objects[1:]
                spec_block = (
                    f"\nSYMBOLIC ONTOLOGY GROUNDING ({fam}):\n"
                    f"- Base block (leave ungrasped on table; query bp = get_object_position('{base}') ONCE at start while unoccluded): {base}\n"
                    f"- Upper blocks to stack bottom-to-top onto {base}: {uppers} "
                    f"(right after grasp(...), call move_to([bp[0], bp[1], bp[2] + k * {priors['release_margin_z']:.3f}]) directly without intermediate lower Z moves for k=1..{len(uppers)})"
                )
            elif fam == "ALIGN_ROW" and len(task_spec.ordered_objects) >= 2:
                spec_block = (
                    f"\nSYMBOLIC ONTOLOGY GROUNDING (ALIGN_ROW):\n"
                    f"- Collinear row order: {task_spec.ordered_objects}; place along clear lane X=0.29 at Y=[-0.22 + k*{priors['row_spacing']:.3f}] (k=0..{len(task_spec.ordered_objects)-1}) and z=0.455"
                )
            elif fam == "DIAGONAL_PAIR" and task_spec.relational_pairs:
                spec_block = (
                    f"\nSYMBOLIC ONTOLOGY GROUNDING (DIAGONAL_PAIR):\n"
                    f"- Moving-to-Reference pairs: {task_spec.relational_pairs}; "
                    f"keep reference blocks in place and move each moving block to [ref_x + {priors['diagonal_delta']:.3f}, ref_y - {priors['diagonal_delta']:.3f}, 0.455]"
                )
            elif fam == "PLACE_INSIDE":
                cname = task_spec.target_container or "brown_box"
                spec_block = (
                    f"\nSYMBOLIC ONTOLOGY GROUNDING (PLACE_INSIDE):\n"
                    f"- Target blocks to place inside '{cname}' (query bp = get_object_position('{cname}')): {task_spec.ordered_objects}\n"
                    f"- Hollow '{cname}' inner cavity half-width is 0.056m (do NOT use +/-0.062m row_spacing or drop multiple 0.05m cubes at the same [bp[0], bp[1]]); "
                    f"place into distinct interior sub-slots ([bp[0]-0.032, bp[1]-0.032, bp[2]+{priors['release_margin_z']:.3f}], [bp[0]+0.032, bp[1]-0.032, bp[2]+{priors['release_margin_z']:.3f}], [bp[0], bp[1]+0.032, bp[2]+{priors['release_margin_z']:.3f}])"
                )
            else:
                spec_block = f"\nCANONICAL TASK FAMILY: {fam}"
        elif canonical_type:
            spec_block = f"\nCANONICAL TASK FAMILY: {canonical_type}"

        prompt = f"""Synthesize flat executable Python lines to accomplish the robotic manipulation task.

AVAILABLE PRIMITIVE API:
- pos = get_object_position("name")  # Returns [x, y, z] center of object from multi-view RGB-D camera
- grasp("name")                      # Approaches, closes parallel-jaw gripper on cube, and lifts
- move_to([x, y, z])                 # Moves end-effector to Cartesian target [x, y, z] in meters
- release()                          # Opens gripper smoothly to place held cube and retracts
- home()                             # Returns robot arm to occlusion-free rest configuration

DETECTED WORKSPACE OBJECTS (use ONLY exact cube names for grasp):
{obj_names}
{objects_str}

SCENE CONTEXT:
{scene_description}

CALIBRATED KINEMATIC PARAMETERS:
- Safe horizontal transit Z: {priors['transit_z']:.3f} m
- Stack release Z offset above lower cube center: +{priors['release_margin_z']:.3f} m (cubes are 0.05m wide)
- Row single-axis center spacing: {priors['row_spacing']:.3f} m
- Diagonal dual-axis (both X and Y) offset: {priors['diagonal_delta']:.3f} m
- Re-localize after home(): {priors['require_home_relocate']}{spec_block}

USER TASK: "{task}"{rules_block}

OBJECT COLOR ALIASES:
- kırmızı/red -> "red_cube", mavi/blue -> "blue_cube", yeşil/green -> "green_cube", sarı/yellow -> "yellow_cube"
- kutu/kap/box -> "brown_box" (fixed hollow container; NEVER pass "brown_box" or "obstacle_wall" to grasp())

SYNTAX CONSTRAINTS:
- Output ONLY flat Python statements using the 5 functions above, variable assignments, list indexing, and +/- arithmetic.
- NO markdown fences, NO comments, NO imports, NO loops/if/def. Always finish with home().
- If physically impossible (non-existent objects, flying, etc.), output ONLY: IMPOSSIBLE: <reason>"""

        if self.client is None:
            return None

        plan = ""
        for attempt in range(5):
            try:
                req_kwargs = dict(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an embodied robot manipulation code synthesizer. "
                                "Emit ONLY flat executable Python statements calling get_object_position, grasp, move_to, release, home. "
                                "No comments, no markdown fences, no control flow statements."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                if "gpt-6" in self.model or "gpt-5" in self.model:
                    req_kwargs["reasoning_effort"] = "low"
                    req_kwargs["max_completion_tokens"] = 1200 + attempt * 250
                else:
                    req_kwargs["temperature"] = 0.0
                    req_kwargs["max_tokens"] = 1200 + attempt * 200
                    if "gpt-oss" in self.model:
                        req_kwargs["reasoning_effort"] = "low"
                    elif "qwen" in self.model:
                        req_kwargs["reasoning_effort"] = "none"
                resp = self.client.chat.completions.create(**req_kwargs)
                raw_content = (resp.choices[0].message.content or "").strip()
                if raw_content:
                    plan = raw_content
                    break
            except Exception as exc:
                err = str(exc).lower()
                if any(tok in err for tok in ("tokens per day", "tpd", "insufficient_quota", "invalid_api_key", "model_not_found")):
                    return None
                if ("429" in err or "rate_limit" in err or "timeout" in err or "503" in err) and attempt < 4:
                    time.sleep(6 * (attempt + 1))
                else:
                    return None
        if not plan:
            return None
        if "</think>" in plan:
            plan = plan.split("</think>")[-1].strip()
        elif "<think>" in plan:
            plan = plan.split("<think>")[-1].strip()
        plan = plan.replace("```python", "").replace("```", "").strip()

        if plan.upper().startswith("IMPOSSIBLE"):
            return None
        return plan
