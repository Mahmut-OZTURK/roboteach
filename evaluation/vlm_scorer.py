# vlm_scorer.py — Multi-view VLM Judge with Causal Failure Taxonomy & Zero Fake-Success Fallback
import os
import re
import time
from groq import Groq
from dotenv import load_dotenv

from calibration.memory import FAILURE_TAXONOMY, FailureDiagnostic
from planner.openai_client import OpenAIChatClient

load_dotenv(override=True)


class VLMScorer:
    """
    Evaluates manipulation outcomes strictly from multi-view RGB camera observations.
    Produces both scalar/boolean task success and structured FailureDiagnostic records
    mapped to the six-class causal failure taxonomy.
    """

    def __init__(
        self,
        camera,
        client=None,
        model: str | None = None,
        max_retries: int = 3,
        min_groq_interval_s: float = 8.5,
    ):
        self.camera = camera
        self.model = model or os.getenv("VLM_JUDGE_MODEL", "gpt-4.1")
        if client is not None:
            self.client = client
        elif self.model.startswith("qwen/") or "gpt-oss" in self.model:
            api_key = (os.getenv("GROQ_API_KEY") or "").strip()
            self.client = Groq(api_key=api_key, max_retries=0, timeout=25.0) if api_key else None
        else:
            api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
            self.client = OpenAIChatClient(api_key=api_key) if api_key else None
        self.scene_critic_model = "openai/gpt-oss-20b"
        self.max_retries = max_retries
        self.min_groq_interval_s = float(min_groq_interval_s)
        self._last_groq_call_ts = 0.0
        self.last_api_latency_ms: float = 0.0

    def _pace_groq(self):
        if isinstance(self.client, Groq) and self.min_groq_interval_s > 0.0:
            elapsed = time.monotonic() - self._last_groq_call_ts
            if elapsed < self.min_groq_interval_s and self._last_groq_call_ts > 0.0:
                time.sleep(self.min_groq_interval_s - elapsed)
            self._last_groq_call_ts = time.monotonic()

    def _chat_create(self, model_name: str, messages: list[dict], max_tokens: int, reasoning_effort: str | None = None):
        self._pace_groq()
        kwargs = {"model": model_name, "messages": messages}
        is_openai_reasoning = "gpt-6" in model_name or "gpt-5" in model_name or model_name.startswith("o")
        if is_openai_reasoning:
            kwargs["max_completion_tokens"] = max_tokens
            kwargs["reasoning_effort"] = reasoning_effort or "low"
            if kwargs["reasoning_effort"] == "none":
                kwargs["temperature"] = 0.0
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = 0.0
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
        t_api = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        self.last_api_latency_ms = round((time.perf_counter() - t_api) * 1000.0, 2)
        return resp

    def analyze_scene(self, images: list | None = None) -> str:
        if self.client is None:
            return "Scene analysis unavailable (no API client)."
        if images is None:
            images = [
                self.camera.capture_b64("front_isometric"),
                self.camera.capture_b64("top"),
                self.camera.capture_b64("side_isometric"),
            ]
        prompt = (
            "Describe the robot manipulation workspace concisely and factually. "
            "List visible cubes, hollow container (brown box), and any purple obstacle wall, "
            "noting existing vertical stacks or occlusions."
        )
        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img['b64']}"}})
        try:
            resp = self._chat_create(
                model_name=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=240,
                reasoning_effort="none",
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return "Scene analysis unavailable."

    def integrity_check(self) -> bool:
        if self.client is None:
            return True
        img = self.camera.capture_b64("top")
        prompt = (
            "Check ONLY for catastrophic physical failures from the overhead workspace view:\n"
            "- Has an object fallen completely off the gray table?\n"
            "Respond ONLY: 'SAFE: True' or 'SAFE: False' followed by 'REASON: <short>'."
        )
        try:
            resp = self._chat_create(
                model_name=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img['b64']}"}},
                        ],
                    }
                ],
                max_tokens=60,
                reasoning_effort="none",
            )
            return "SAFE: TRUE" in resp.choices[0].message.content.upper()
        except Exception:
            return True

    def evaluate(
        self,
        task: str,
        perceived_details: dict | None = None,
        prefer_multimodal: bool = True,
        allow_pointcloud_fallback: bool = False,
    ) -> dict:
        if self.client is None:
            return {
                "success": False,
                "score": 0.0,
                "evaluation_error": True,
                "judge_mode": "none",
                "failure_code": "F_EXEC_SYNTAX",
                "issue": "VLM client not configured",
                "instruction": None,
                "diagnostic": None,
                "api_latency_ms": 0.0,
            }

        self.last_api_latency_ms = 0.0
        codes_desc = "\n".join(f"- {k}: {v}" for k, v in FAILURE_TAXONOMY.items())
        parsed = None
        judge_mode = "multimodal_vlm"
        last_err = None
        obs_summary_block = ""
        if perceived_details:
            obs_items = []
            for name, info in perceived_details.items():
                pos = [round(float(v), 3) for v in info["position"]]
                if name == "brown_box":
                    obs_items.append(f"brown_box: center_xyz={pos}, inner_opening_half_extent=0.056m, rim_top_z=0.490m")
                else:
                    obs_items.append(f"{name}: center_xyz={pos}")
            obs_summary_block = "\nPERCEIVED 3D SUMMARY (meters, table_top_z=0.420, cube_half=0.025m): " + "; ".join(obs_items)

        if prefer_multimodal:
            if hasattr(self.camera, "capture_multiview_strip_b64"):
                strip = self.camera.capture_multiview_strip_b64(save_dir=None)
                image_attachments = [
                    {"type": "image_url", "image_url": {"url": f"data:{strip['mime']};base64,{strip['b64']}"}}
                ]
            else:
                img_iso = self.camera.capture_b64("front_isometric", "captures")
                img_top = self.camera.capture_b64("top", "captures")
                img_side = self.camera.capture_b64("side_isometric", "captures")
                image_attachments = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_iso['b64']}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_top['b64']}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_side['b64']}"}},
                ]

            vlm_prompt = f"""Evaluate the FINAL workspace state of this robotic manipulation task using the three synchronized camera views (left: front isometric, center: overhead top, right: side isometric) AND the perceived 3D center coordinates (if given).
COLOR/OBJECT LEGEND (Turkish -> workspace object): kırmızı=red cube, mavi=blue cube, yeşil=green cube, sarı=yellow cube, kahverengi kutu/açık kutu=hollow brown box, duvar/engel=purple obstacle wall.
WORKSPACE GEOMETRY FACTS: Table surface is at z=0.420m (cubes resting flat on table have center z~0.445m). Cubes are 0.050m (half=0.025m). A block stacked on another has vertical gap dz>=0.036m and horizontal dx,dy<0.025m. Hollow brown_box has center xy and inner cavity half-extent ~0.056m; cubes inside have |dx|<0.056m and |dy|<0.056m relative to brown_box AND z between 0.43m and 0.52m (above z=0.50m perched on rim fails).
TASK: "{task}"{obs_summary_block}
JUDGING CRITERIA (Physical settling tolerance: slight tilt or minor XY offset <0.030m counts as SUCCESS when the requested spatial relation holds):
(1) Ordered stack: requested bottom-to-top color sequence upright on table (each upper block z higher than block below by >=0.036m, aligned within 0.025m); base cube stays at bottom (~0.445m). Ignore intermediate motions; judge ONLY the final state.
(2) Row ("yan yana"): all 4 cubes resting side-by-side on table (z~0.445m, none stacked) in a roughly collinear line along X or Y without piling.
(3) Diagonal ("çaprazına"): target cube offset along BOTH X and Y (0.04m..0.11m) relative to reference cube, both resting flat on table (z~0.445m). Ignore blocks not named in the diagonal pair unless disturbed.
(4) Container ("kutunun içine"): ALL requested target cubes must be seated inside the brown_box cavity (verify from overhead top view AND coordinates: |x-box_x|<0.056m, |y-box_y|<0.056m). If even 1 target cube sits OUTSIDE the box, on the rim, or on the table, VISUAL_SUCCESS MUST be False and FAILURE_CODE MUST be F_CONTAINER_RIM (NEVER F_XY_ALIGN) with partial score.
(5) Swap / Unstack / Obstacle ("ama ... zaten ..." / "engelin"): in Turkish commands like "X ile Y'yi üst üste koy, ama Z zaten A'nın üzerinde", the clause after "ama" describes the INITIAL state to undo so that the final state achieves the goal BEFORE "ama" (e.g., X and Y stacked upright on table, and the other block safely placed on table z~0.445m); verify the requested final stack AND that no cube fell off the table (z<0.35m).
FAILURE CODES (use None if successful):
{codes_desc}
Reply strictly with these 6 lines (put verdict lines FIRST, keep REASONING to 1 short sentence):
VISUAL_SUCCESS: True or False
VISUAL_SCORE: <0.00 to 1.00>
FAILURE_CODE: <code or None>
ISSUE: <short cause or None>
LEARNED_INSTRUCTION: <1 short geometric fix or None>
REASONING: <1 concise factual sentence citing observed positions/views>"""

            content = [{"type": "text", "text": vlm_prompt}] + image_attachments
            is_reasoning = "gpt-6" in self.model or "gpt-5" in self.model or self.model.startswith("o")
            eff = "low" if is_reasoning else ("none" if "qwen" in self.model else None)
            tok_budget = 700 if is_reasoning else (240 if "qwen" in self.model else 300)
            attempts = max(1, self.max_retries)
            for attempt in range(attempts):
                try:
                    resp = self._chat_create(
                        model_name=self.model,
                        messages=[{"role": "user", "content": content}],
                        max_tokens=tok_budget,
                        reasoning_effort=eff,
                    )
                    txt = (resp.choices[0].message.content or "").strip()
                    if txt:
                        cand = self._parse(txt)
                        parsed = cand
                        if not cand.get("parse_error"):
                            break
                        last_err = "Missing or invalid VISUAL_SUCCESS in VLM output"
                except Exception as exc:
                    last_err = str(exc)
                    low_err = last_err.lower()
                    if "tokens per day" in low_err or "tpd" in low_err or "insufficient_quota" in low_err:
                        break
                    if any(k in low_err for k in ("429", "rate_limit", "timeout", "503", "502", "connection")) and attempt + 1 < attempts:
                        m_wait = re.search(r"try again in\s+(\d+\.?\d*)s", low_err)
                        sleep_s = (float(m_wait.group(1)) + 1.5) if m_wait else (3.0 * (attempt + 1))
                        time.sleep(min(sleep_s, 65.0))
                    else:
                        break

        if (parsed is None or parsed.get("parse_error")) and (not prefer_multimodal or allow_pointcloud_fallback) and perceived_details is not None:
            judge_mode = "rgbd_pointcloud_critic"
            obs_lines = []
            for name, info in perceived_details.items():
                pos = [round(float(v), 3) for v in info["position"]]
                zt = round(float(info.get("z_top", pos[2] + 0.025)), 3)
                occ = ", occluded_from_top=True" if info.get("occluded_from_top") else ""
                obs_lines.append(f"  - {name}: center_xyz={pos}m, z_top={zt}m{occ}")
            obs_block = "\n".join(obs_lines) if obs_lines else "  (no objects detected in workspace)"
            critic_prompt = f"""Evaluate whether the perceived 3D RGB-D camera point-cloud scene satisfies the manipulation command.
Table surface Z = 0.420m; cube width = 0.050m (resting table center z ≈ 0.445m, stacked layer delta_z ≈ +0.050m [>= +0.036m], max stack xy misalignment <= 0.032m for unoccluded top cubes or <= 0.055m when occluded_from_top=True due to lateral depth-ray uncertainty).
Collinear row ("yan yana"): all 4 cubes resting on table (z in [0.420, 0.470]m), cross-axis spread <= 0.032m, adjacent center gaps in [0.046, 0.095]m.
Diagonal pair ("çaprazına"): both cubes resting on table with |dx| in [0.035, 0.115]m AND |dy| in [0.035, 0.115]m.
Hollow brown_box cavity center = [0.250, 0.230]m, interior half-width = 0.068m (x in [0.182, 0.318], y in [0.162, 0.298]), interior z in [0.425, 0.515]m (note: brown_box walls may partially occlude interior cubes from lateral cameras).
Color map: kırmızı=red_cube, mavi=blue_cube, yeşil=green_cube, sarı=yellow_cube, kahverengi kutu=brown_box.

TASK: "{task}"

CAMERA-DEPROJECTED 3D OBJECT STATES:
{obs_block}

FAILURE CODES (use None if successful):
{codes_desc}

Reply concisely in 6 lines ONLY (do NOT over-elaborate):
REASONING: <1 factual sentence>
VISUAL_SUCCESS: True or False
VISUAL_SCORE: <0.00 to 1.00>
FAILURE_CODE: <code or None>
ISSUE: <concise cause or None>
LEARNED_INSTRUCTION: <1 actionable geometric correction or None>"""
            for attempt in range(self.max_retries):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.scene_critic_model,
                        reasoning_effort="low",
                        messages=[{"role": "user", "content": critic_prompt}],
                        temperature=0.0,
                        max_tokens=900 + attempt * 250,
                    )
                    txt = (resp.choices[0].message.content or "").strip()
                    if txt:
                        cand = self._parse(txt)
                        parsed = cand
                        if not cand.get("parse_error"):
                            break
                        last_err = "Missing or invalid VISUAL_SUCCESS in critic output"
                except Exception as exc:
                    last_err = str(exc)
                    if attempt + 1 < self.max_retries:
                        time.sleep(5.0 * (attempt + 1))

        if parsed is None or parsed.get("parse_error"):
            return {
                "success": False,
                "score": 0.0,
                "evaluation_error": True,
                "judge_mode": "error",
                "failure_code": "F_EXEC_SYNTAX",
                "issue": f"VLM evaluation failed: {last_err or 'Missing VISUAL_SUCCESS'}",
                "instruction": None,
                "diagnostic": None,
                "api_latency_ms": self.last_api_latency_ms,
            }

        diag = None
        if not parsed["visual_success"]:
            diag = self._build_diagnostic(
                parsed["failure_code"], parsed["issue"], parsed["instruction"]
            )

        return {
            "success": parsed["visual_success"],
            "score": parsed["visual_score"],
            "evaluation_error": False,
            "judge_mode": judge_mode,
            "failure_code": parsed["failure_code"],
            "issue": parsed["issue"],
            "instruction": parsed["instruction"],
            "reasoning": parsed["reasoning"],
            "diagnostic": diag,
            "api_latency_ms": self.last_api_latency_ms,
        }

    def _build_diagnostic(self, code: str, issue: str, instruction: str | None) -> FailureDiagnostic:
        norm_code = code if code in FAILURE_TAXONOMY else self._infer_code_from_text(f"{issue} {instruction or ''}")
        updates = {}
        rule = ""

        if norm_code == "F_COLLISION":
            updates["transit_z"] = 0.68
            updates["release_margin_z"] = 0.052
            rule = "Lift end-effector to safe transit height z >= 0.68m before any horizontal XY motion and release at +0.052m per stacked layer."
        elif norm_code == "F_Z_DROP":
            updates["transit_z"] = 0.68
            updates["release_margin_z"] = 0.052
            rule = "Descend to target_z + 0.052m above the supporting block center before calling release() to prevent impact toppling."
        elif norm_code == "F_CONTAINER_RIM":
            updates["transit_z"] = 0.68
            updates["release_margin_z"] = 0.085
            rule = (
                "Place cubes into distinct interior sub-slots around brown_box center "
                "([bx-0.032, by-0.032, 0.525], [bx+0.032, by-0.032, 0.525], [bx, by+0.032, 0.525]) "
                "before release() so cubes rest side-by-side inside the cavity without piling over the rim."
            )
        elif norm_code == "F_OCCLUSION_STALE":
            updates["require_home_relocate"] = True
            updates["release_margin_z"] = 0.052
            rule = "Call home() after each release() and re-query get_object_position() before grasping or stacking the next block."
        elif norm_code == "F_ORDER":
            updates["release_margin_z"] = 0.052
            updates["transit_z"] = 0.68
            rule = "Leave the bottom-most base block ungrasped on the table and stack upper blocks strictly in bottom-to-top order."
        elif norm_code == "F_XY_ALIGN":
            updates["diagonal_delta"] = 0.062
            updates["row_spacing"] = 0.062
            updates["release_margin_z"] = 0.052
            updates["transit_z"] = 0.68
            rule = "Use 0.062m center spacing along one axis for collinear rows, +0.062m/-0.062m on BOTH X and Y axes for diagonal pairs, and +0.052m per layer when stacking."
        elif instruction and instruction != "None":
            rule = instruction

        return FailureDiagnostic(
            failure_code=norm_code,
            summary=issue or "Visual manipulation failure",
            param_updates=updates,
            symbolic_rule=rule,
        )

    @staticmethod
    def _infer_code_from_text(text: str) -> str:
        low = text.lower()
        if any(w in low for w in ("collid", "knock", "wall", "obstacle", "fell off")):
            return "F_COLLISION"
        if any(w in low for w in ("rim", "box", "container", "outside")):
            return "F_CONTAINER_RIM"
        if any(w in low for w in ("order", "base", "bottom", "sequence", "wrong color")):
            return "F_ORDER"
        if any(w in low for w in ("drop", "bounc", "tilt", "toppl", "height")):
            return "F_Z_DROP"
        if any(w in low for w in ("occlu", "camera", "stale")):
            return "F_OCCLUSION_STALE"
        return "F_XY_ALIGN"

    @staticmethod
    def _parse_bool_verdict(val: str) -> tuple[bool, bool]:
        """Parses VISUAL_SUCCESS value, handling negated phrases; returns (verdict, valid)."""
        low = val.strip().lower()
        if not low:
            return False, False
        if re.search(r"\b(?:not\s+true|not\s+yes|not\s+successful|unsuccessful|false)\b", low):
            return False, True
        if re.match(r"^\s*no\b", low) or (re.search(r"\bno\b", low) and not re.match(r"^\s*(?:true|yes)\b", low)):
            return False, True
        if re.search(r"\b(?:true|yes)\b", low):
            return True, True
        return False, False

    @staticmethod
    def _parse(text: str) -> dict:
        out = {
            "visual_success": False,
            "visual_score": 0.0,
            "failure_code": "F_XY_ALIGN",
            "issue": "Parse error",
            "instruction": None,
            "reasoning": "",
            "parse_error": True,
        }
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        clean = text.replace("**", "").replace("###", "").replace("##", "")
        found = False
        has_visual_success = False
        for raw in clean.splitlines():
            line = raw.strip()
            if not line or ":" not in line:
                continue
            head, val = line.split(":", 1)
            key = head.strip().upper()
            val = val.strip()
            if key in ("REASONING", "ANALYSIS", "EXPLANATION"):
                out["reasoning"] = val
                found = True
            elif key in ("VISUAL_SUCCESS", "SUCCESS"):
                verdict, is_valid = VLMScorer._parse_bool_verdict(val)
                if is_valid:
                    out["visual_success"] = verdict
                    has_visual_success = True
                found = True
            elif key in ("VISUAL_SCORE", "SCORE"):
                m = re.search(r"(\d+\.?\d*)", val)
                if m:
                    out["visual_score"] = max(0.0, min(1.0, float(m.group(1))))
                    found = True
            elif key == "FAILURE_CODE":
                out["failure_code"] = val.split()[0].strip() if val else "None"
                found = True
            elif key in ("ISSUE", "ERROR"):
                out["issue"] = val
                found = True
            elif key in ("LEARNED_INSTRUCTION", "TIP", "ADVICE"):
                out["instruction"] = None if val.upper() == "NONE" else val
                found = True

        out["parse_error"] = not has_visual_success
        if out["parse_error"]:
            out["visual_success"] = False
            out["visual_score"] = 0.0
            if out["issue"] in ("Parse error", "None"):
                out["issue"] = "Parse error: missing VISUAL_SUCCESS"
        elif found and out["issue"] == "Parse error":
            out["issue"] = "None"
        if out["visual_success"] and not out["parse_error"]:
            out["failure_code"] = "None"
        return out
