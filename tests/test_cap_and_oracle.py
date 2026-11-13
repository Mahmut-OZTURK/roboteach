import pybullet as p

from simulation.env import RoboTeachEnv
from calibration.ontology import CanonicalTaskOntology
from evaluation.oracle import GroundTruthOracle, compute_judge_agreement
from evaluation.vlm_scorer import VLMScorer
from planner.cap import CaP


class DummySkills:
    def __init__(self):
        self.calls = []

    def move_to(self, pos):
        self.calls.append(("move_to", pos))
        return pos

    def grasp(self, name):
        self.calls.append(("grasp", name))
        return True

    def release(self):
        self.calls.append(("release",))
        return True

    def home(self):
        self.calls.append(("home",))

    def get_object_position(self, name):
        return [0.15, 0.0, 0.445]


def test_cap_ast_guard_blocks_unsafe_execution():
    skills = DummySkills()
    cap = CaP(skills)
    assert cap.execute('import os\nos.system("echo bad")') is False
    assert cap.execute('__import__("sys")') is False
    assert cap.execute('p = get_object_position("red_cube")\ngrasp("red_cube")\nmove_to([p[0], p[1], 0.65])\nrelease()\nhome()') is True
    assert len(skills.calls) == 4


def test_vlm_scorer_never_fakes_success_on_api_error():
    class BrokenClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("Simulated API outage")

    class DummyCamera:
        def capture_b64(self, view_name, save_dir=None):
            return {"b64": "AA==", "view": view_name}

    scorer = VLMScorer(DummyCamera(), client=BrokenClient(), max_retries=1)
    res = scorer.evaluate("stack red on blue")
    assert res["success"] is False
    assert res["evaluation_error"] is True
    assert res["score"] == 0.0


def test_ground_truth_oracle_and_cohens_kappa():
    env = RoboTeachEnv(headless=True, seed=5, include_container=True)
    try:
        ont = CanonicalTaskOntology(llm_client=None)
        oracle = GroundTruthOracle(env)
        spec = ont.parse("kutuları kule gibi üst üste diz: en altta mavi üstünde kırmızı üstünde yeşil en üstte sarı")

        # Initially scattered -> must be False
        r0 = oracle.evaluate(spec)
        assert r0["success"] is False

        # Arrange exact 4-cube ordered tower on table
        order = ["blue_cube", "red_cube", "green_cube", "yellow_cube"]
        for idx, cname in enumerate(order):
            p.resetBasePositionAndOrientation(
                env.objects[cname], [0.16, 0.0, 0.445 + idx * 0.050], [0, 0, 0, 1]
            )
            p.resetBaseVelocity(env.objects[cname], [0, 0, 0], [0, 0, 0])
        for _ in range(60):
            p.stepSimulation()

        r1 = oracle.evaluate(spec)
        assert r1["success"] is True
        assert r1["score"] >= 0.95
    finally:
        env.close()

    metrics = compute_judge_agreement(
        oracle_labels=[True, True, False, False, True, False, True, False],
        vlm_labels=[True, True, False, True, True, False, True, False],
    )
    assert 0.70 <= metrics["accuracy"] <= 1.0
    assert metrics["cohen_kappa"] > 0.45

