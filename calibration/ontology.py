# ontology.py — Deterministic Canonical Spatial Task Ontology & Precondition Analyzer
from dataclasses import dataclass, field
import json
import re


COLOR_ALIASES = {
    "red_cube": ["red_cube", "red", "kırmızı", "kirmizi"],
    "blue_cube": ["blue_cube", "blue", "mavi"],
    "green_cube": ["green_cube", "green", "yeşil", "yesil"],
    "yellow_cube": ["yellow_cube", "yellow", "sarı", "sari"],
}

ALL_CUBES = ["red_cube", "blue_cube", "green_cube", "yellow_cube"]


@dataclass
class TaskSpec:
    raw_task: str
    canonical_type: str
    ordered_objects: list[str] = field(default_factory=list)
    relational_pairs: list[tuple[str, str]] = field(default_factory=list)  # (moving, reference)
    target_container: str | None = None
    initial_stacks: list[list[str]] = field(default_factory=list)  # [[top, bottom]]
    has_obstacle: bool = False

    def to_scene_config(self) -> dict:
        return {
            "stacks": [list(p) for p in self.initial_stacks],
            "inside_box": [],
            "has_obstacle": self.has_obstacle,
        }


class CanonicalTaskOntology:
    """
    Resolves free-form Turkish and English manipulation instructions into a canonical
    spatial operator family, eradicating synonym split-brain (e.g., put_diagonally vs place_diagonally,
    build_tower vs stack_in_order) while extracting object roles and initial preconditions.
    """

    CANONICAL_FAMILIES = (
        "STACK_ORDERED",
        "ALIGN_ROW",
        "DIAGONAL_PAIR",
        "PLACE_INSIDE",
        "UNSTACK_SWAP",
        "OBSTACLE_CLEARANCE",
    )

    def __init__(self, llm_client=None, model: str = "qwen/qwen3.8-27b"):
        self.llm_client = llm_client
        self.model = model

    def _extract_ordered_mentions(self, text: str) -> list[str]:
        low = text.lower()
        hits: list[tuple[int, str]] = []
        for canonical, aliases in COLOR_ALIASES.items():
            best_idx = None
            for alias in aliases:
                m = re.search(rf"\b{re.escape(alias)}", low)
                if m and (best_idx is None or m.start() < best_idx):
                    best_idx = m.start()
            if best_idx is not None:
                hits.append((best_idx, canonical))
        hits.sort(key=lambda item: item[0])
        return [c for _, c in hits]

    def _detect_initial_stacks(self, low: str) -> list[list[str]]:
        stacks: list[list[str]] = []
        pre_match = re.search(
            r"\b(?:ama|fakat|ancak|oysa|lakin|but|while|whereas|although|even\s+though|though|however)\b[,:\s]*(.+)$",
            low,
        )
        if re.match(r"^(?:although|even\s+though|though|given\s+that|since|currently|initially|başlangıçta|baslangicta|şu anda|su anda)\b", low) and re.search(r"[,;]", low):
            clause = re.split(r"[,;]", low, maxsplit=1)[0]
        elif pre_match:
            clause = pre_match.group(1)
        else:
            clause = low
        pre_kw = ("zaten", "already", "başlangıçta", "baslangicta", "starts on", "starts atop", "currently", "initially", "halihazırda", "halihazirda", "şu anda", "su anda")
        if not any(k in clause for k in pre_kw):
            return stacks

        mentions = self._extract_ordered_mentions(clause)
        if len(mentions) >= 2:
            for c_a in mentions:
                for c_b in mentions:
                    if c_a == c_b:
                        continue
                    aliases_a = COLOR_ALIASES[c_a]
                    aliases_b = COLOR_ALIASES[c_b]
                    for aa in aliases_a:
                        for ab in aliases_b:
                            if re.search(
                                rf"{aa}.*(?:already|zaten|currently|initially|halihazırda|halihazirda|şu anda|su anda|starts).*{ab}.*(?:on|atop|above|üst|uzer)",
                                clause,
                            ) or re.search(
                                rf"{aa}\s+(?:zaten\s+)?{ab}\w*\s+(?:üstünde|üzerinde|ustunde|uzerinde)", clause
                            ):
                                return [[c_a, c_b]]
                            if re.search(rf"{ab}\w*\s+(?:üstünde|üzerinde|ustunde|uzerinde).*{aa}", clause):
                                return [[c_a, c_b]]
            stacks.append([mentions[0], mentions[1]])
        return stacks

    def _extract_diagonal_pairs(self, low: str, mentions: list[str]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        clauses = re.split(r"[,;]|(?:\bve\b|\band\b)", low)
        for cl in clauses:
            c_mentions = self._extract_ordered_mentions(cl)
            if len(c_mentions) >= 2:
                cap_idx = -1
                for tok in ("çapraz", "capraz", "köşegen", "kosegen"):
                    pos = cl.find(tok)
                    if pos != -1 and (cap_idx == -1 or pos < cap_idx):
                        cap_idx = pos
                if cap_idx != -1:
                    idx0 = min((cl.find(a) for a in COLOR_ALIASES[c_mentions[0]] if cl.find(a) != -1), default=0)
                    idx1 = min((cl.find(a) for a in COLOR_ALIASES[c_mentions[1]] if cl.find(a) != -1), default=0)
                    # "maviyi kırmızının çaprazına" (idx0 < idx1 < cap_idx): first is moving, second is ref
                    if idx0 < idx1 < cap_idx:
                        pairs.append((c_mentions[0], c_mentions[1]))
                    else:
                        pairs.append((c_mentions[1], c_mentions[0]))
                else:
                    pairs.append((c_mentions[0], c_mentions[1]))
        if not pairs and len(mentions) >= 2:
            if any(t in low for t in ("çapraz", "capraz", "köşegen", "kosegen")):
                for i in range(0, len(mentions) - 1, 2):
                    pairs.append((mentions[i + 1], mentions[i]))
            else:
                for i in range(0, len(mentions) - 1, 2):
                    pairs.append((mentions[i], mentions[i + 1]))
        return pairs

    def parse(self, task: str) -> TaskSpec:
        low = task.strip().lower()
        initial_stacks = self._detect_initial_stacks(low)

        # Strip precondition clause when parsing the goal clause
        pre_split_re = r"\b(?:ama|fakat|ancak|oysa|lakin|but|while|whereas|although|even\s+though|though|however)\b"
        lead_pre_re = r"^(?:although|even\s+though|though|given\s+that|since|currently|initially|başlangıçta|baslangicta|şu anda|su anda)\b"
        if re.match(lead_pre_re, low) and re.search(r"[,;]", low):
            goal_clause = re.split(r"[,;]", low, maxsplit=1)[1]
        elif re.search(pre_split_re, low):
            goal_clause = re.split(pre_split_re, low, maxsplit=1)[0]
        else:
            goal_clause = low
        mentions = self._extract_ordered_mentions(goal_clause)

        has_obstacle = any(
            w in low for w in (
                "engel", "duvar", "barrier", "obstacle", "wall", "aşırarak", "over the wall",
                "bariyer", "aşarak", "asarak", "üzerinden atlat", "üzerinden geçir", "divider", "partition", "hurdle",
            )
        )
        is_inside = any(
            w in goal_clause for w in (
                "içine", "icine", "inside", "into the box", "kutuya", "kutunun içine", "kaba",
                "into the container", "in the box", "in the container", "within the box",
                "inside the brown", "into brown", "içerisine", "icerisine", "kabın", "hazne",
            )
        ) or bool(re.search(r"\b(?:in|into|inside|within)\s+(?:the\s+)?(?:\w+\s+)?(?:box|container|brown_box)\b", goal_clause))
        is_diagonal = any(w in goal_clause for w in ("çapraz", "capraz", "köşegen", "kosegen", "diagonal", "diagonally", "crosswise"))
        is_row = any(
            w in goal_clause
            for w in (
                "yan yana", "yanyana", "yanına", "yanina", "beside", "next to", "in a row", "sıra halinde", "line up",
                "side by side", "side-by-side", "in a line", "collinear", "a row", "tek sıra", "tek sira", "hizala", "hizada", "sıraya", "siraya", "doğrusal", "dogrusal",
            )
        ) or bool(re.search(r"\b(?:row|line)\b", goal_clause))
        is_stack = any(
            w in goal_clause
            for w in (
                "üst üste", "üstüne", "üzerine", "ustune", "uzerine", "kule", "istifle",
                "stack", "on top", "tower", "pile", "yığ", "yig", "tepesine", "atop",
            )
        )

        if initial_stacks and (is_stack or re.search(r"\b(?:on|onto|swap|reverse)\b", goal_clause) or "yer değiştir" in goal_clause):
            canonical = "UNSTACK_SWAP"
            # Goal: "put blue on red" -> bottom is red, top is blue; or Turkish "maviyi kırmızının üstüne koy" -> moving=blue, base=red
            ordered = self._resolve_two_object_stack_order(goal_clause, mentions)
            return TaskSpec(
                raw_task=task,
                canonical_type=canonical,
                ordered_objects=ordered,
                initial_stacks=initial_stacks,
                has_obstacle=has_obstacle,
            )

        if has_obstacle:
            ordered = (
                self._resolve_two_object_stack_order(goal_clause, mentions)
                if len(mentions) == 2
                else (mentions if mentions else list(ALL_CUBES[:2]))
            )
            return TaskSpec(
                raw_task=task,
                canonical_type="OBSTACLE_CLEARANCE",
                ordered_objects=ordered,
                initial_stacks=initial_stacks,
                has_obstacle=True,
            )

        if is_inside:
            targets = mentions if mentions else list(ALL_CUBES)
            return TaskSpec(
                raw_task=task,
                canonical_type="PLACE_INSIDE",
                ordered_objects=targets,
                target_container="brown_box",
                initial_stacks=initial_stacks,
                has_obstacle=has_obstacle,
            )

        if is_diagonal:
            pairs = self._extract_diagonal_pairs(goal_clause, mentions)
            return TaskSpec(
                raw_task=task,
                canonical_type="DIAGONAL_PAIR",
                ordered_objects=mentions if mentions else list(ALL_CUBES),
                relational_pairs=pairs,
                initial_stacks=initial_stacks,
                has_obstacle=has_obstacle,
            )

        if is_row and not is_stack:
            targets = mentions if len(mentions) >= 2 else list(ALL_CUBES)
            return TaskSpec(
                raw_task=task,
                canonical_type="ALIGN_ROW",
                ordered_objects=targets,
                initial_stacks=initial_stacks,
                has_obstacle=has_obstacle,
            )

        if is_stack:
            if len(mentions) == 2:
                ordered = self._resolve_two_object_stack_order(goal_clause, mentions)
            elif len(mentions) >= 3:
                ordered = mentions
            else:
                ordered = list(ALL_CUBES)
            return TaskSpec(
                raw_task=task,
                canonical_type="STACK_ORDERED",
                ordered_objects=ordered,
                initial_stacks=initial_stacks,
                has_obstacle=has_obstacle,
            )

        # Optional LLM fallback strictly constrained to canonical set if client available
        if self.llm_client is not None:
            try:
                resp = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Map this robot manipulation command to ONE canonical family from "
                                f"{list(self.CANONICAL_FAMILIES)}. Respond ONLY with JSON: "
                                f'{{"family": "..."}}\nCommand: "{task}"'
                            ),
                        }
                    ],
                    temperature=0.0,
                    max_tokens=40,
                )
                txt = resp.choices[0].message.content.strip()
                m = re.search(r"\{.*\}", txt, re.DOTALL)
                if m:
                    fam = json.loads(m.group(0)).get("family", "")
                    if fam in self.CANONICAL_FAMILIES:
                        return TaskSpec(
                            raw_task=task,
                            canonical_type=fam,
                            ordered_objects=mentions or list(ALL_CUBES),
                            initial_stacks=initial_stacks,
                            has_obstacle=has_obstacle,
                        )
            except Exception:
                pass

        return TaskSpec(
            raw_task=task,
            canonical_type="STACK_ORDERED",
            ordered_objects=mentions or list(ALL_CUBES),
            initial_stacks=initial_stacks,
            has_obstacle=has_obstacle,
        )

    def _resolve_two_object_stack_order(self, clause: str, mentions: list[str]) -> list[str]:
        """Returns [bottom_cube, top_cube] for a 2-object stacking phrase."""
        if len(mentions) < 2:
            return mentions or ["blue_cube", "red_cube"]
        c0, c1 = mentions[0], mentions[1]
        # If explicitly phrased "en altta X üstünde Y" or "bottom X top Y", c0 is already bottom
        if any(p in clause for p in ("en altta", "altta", "bottom", "tabanda", "sırasıyla", "in order")):
            return [c0, c1]
        # Otherwise "put A on top of B" / "A küpünü B'nin üstüne koy": c0 is moving (top), c1 is support (bottom)
        return [c1, c0]

    def decompose_checkpoints(self, spec: TaskSpec) -> list[str]:
        """Produces deterministic, verifiable sequential sub-goals for multi-step execution."""
        if spec.canonical_type == "UNSTACK_SWAP" and spec.initial_stacks:
            top_init, bot_init = spec.initial_stacks[0]
            steps = [
                f"unstack {top_init} from {bot_init} and place {top_init} on the unoccupied table coordinate [0.14, -0.29, 0.475]"
            ]
            if len(spec.ordered_objects) >= 2:
                steps.append(f"stack {spec.ordered_objects[1]} directly on top of {spec.ordered_objects[0]} (z + 0.052)")
            return steps
        return [spec.raw_task]
