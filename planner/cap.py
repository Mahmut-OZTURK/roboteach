# cap.py — AST-Validated Code-as-Policies Executor with Stepwise Physical Integrity Monitoring
import ast
import time
import pybullet as p


ALLOWED_AST_NODES = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Subscript,
    ast.Slice,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
    ast.Call,
)


class CaP:
    """
    Executes synthesized Python manipulation policies inside an AST-verified sandbox.
    Enforces:
      1. Static AST safety verification (blocks imports, attribute access, loops, arbitrary builtins)
      2. Stepwise post-grasp contact verification & post-release physical settling + integrity check
    """

    def __init__(self, skills, physics_client_id: int = 0):
        self.skills = skills
        self.client = physics_client_id
        self.last_error: str | None = None
        self.last_failure_code: str | None = None
        self.sandbox = {
            "move_to": self.skills.move_to,
            "grasp": self.skills.grasp,
            "release": self.skills.release,
            "home": self.skills.home,
            "get_object_position": self.skills.get_object_position,
            "round": round,
            "max": max,
            "min": min,
        }

    def _clean_plan(self, plan: str) -> list[str]:
        clean = []
        for raw in plan.strip().splitlines():
            s = raw.strip()
            if not s or s.startswith("#") or "```" in s:
                continue
            clean.append(s)
        return clean

    def _validate_ast(self, source: str) -> bool:
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError as exc:
            self.last_error = f"SyntaxError: {exc}"
            self.last_failure_code = "F_EXEC_SYNTAX"
            return False

        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_AST_NODES):
                self.last_error = f"Disallowed AST construct: {type(node).__name__}"
                self.last_failure_code = "F_EXEC_SYNTAX"
                return False
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in self.sandbox:
                    fname = getattr(node.func, "id", repr(node.func))
                    self.last_error = f"Unauthorized function call: {fname}"
                    self.last_failure_code = "F_EXEC_SYNTAX"
                    return False
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                self.last_error = f"Disallowed dunder identifier: {node.id}"
                self.last_failure_code = "F_EXEC_SYNTAX"
                return False
        return True

    def execute(
        self, plan: str, scorer=None, enable_step_integrity: bool = False
    ) -> bool:
        self.last_error = None
        self.last_failure_code = None
        lines = self._clean_plan(plan)
        if not lines:
            self.last_error = "Empty plan"
            self.last_failure_code = "F_EXEC_SYNTAX"
            return False

        joined = "\n".join(lines)
        if not self._validate_ast(joined):
            return False

        scope = dict(self.sandbox)
        realtime = getattr(self.skills, "realtime", False)

        for line in lines:
            try:
                stmt = ast.parse(line, mode="exec").body[0]
                is_direct_grasp = (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id == "grasp"
                )
                if is_direct_grasp:
                    ok = eval(compile(ast.Expression(stmt.value), "<cap>", "eval"), {"__builtins__": {}}, scope)
                    if ok is False:
                        self.last_error = f"Grasp failed at step: {line}"
                        self.last_failure_code = "F_COLLISION"
                        return False
                else:
                    exec(compile(ast.Module(body=[stmt], type_ignores=[]), "<cap>", "exec"), {"__builtins__": {}}, scope)

                if "release(" in line:
                    if p.isConnected(self.client):
                        for _ in range(90):
                            p.stepSimulation(physicsClientId=self.client)
                            if realtime:
                                time.sleep(1.0 / 240.0)
                    if enable_step_integrity and scorer is not None:
                        if not scorer.integrity_check():
                            self.last_error = f"Catastrophic workspace integrity failure after: {line}"
                            self.last_failure_code = "F_COLLISION"
                            return False
            except Exception as exc:
                self.last_error = f"Runtime error at '{line}': {exc}"
                self.last_failure_code = "F_EXEC_SYNTAX"
                return False

        return True
