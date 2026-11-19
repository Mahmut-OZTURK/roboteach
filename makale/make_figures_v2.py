"""Builds the simplified English figures for the revised manuscript (makale/figures_v2)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "makale", "figures_v2")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 14})
INK = "#1f2937"
BLUE = "#2563eb"
GREY = "#6b7280"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, name + ".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------- Figure 1: architecture flowchart ----------------
def fig1():
    steps = [
        ("Command", "Turkish / English"),
        ("Task parsing", "6 task types"),
        ("Perception", "4 RGB-D cameras"),
        ("Code\ngeneration", "LLM"),
        ("Check and\nexecute", "robot arm"),
        ("VLM judge", "success?"),
    ]
    fig, ax = plt.subplots(figsize=(16, 4.4))
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    w, h, y = 2.2, 1.5, 2.3
    xs = [0.2 + i * 2.7 for i in range(len(steps))]
    for x, (title, sub) in zip(xs, steps):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#eff6ff", ec=BLUE, lw=1.6))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=14, weight="bold", color=INK, linespacing=1.0)
        ax.text(x + w / 2, y + h * 0.2, sub, ha="center", va="center", fontsize=12, color=GREY)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((a + w, y + h / 2), (b, y + h / 2), arrowstyle="-|>", mutation_scale=16, lw=1.6, color=INK))
    # calibration box below the judge, loop back to code generation
    cx, cy = xs[4] + 0.6, 0.2
    ax.add_patch(FancyBboxPatch((cx, cy), 3.2, 1.2, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#fef3c7", ec="#d97706", lw=1.6))
    ax.text(cx + 1.6, cy + 0.8, "Calibration", ha="center", va="center", fontsize=14, weight="bold", color=INK)
    ax.text(cx + 1.6, cy + 0.36, "update motion settings", ha="center", va="center", fontsize=12, color=GREY)
    ax.add_patch(FancyArrowPatch((xs[5] + w / 2, y), (cx + 2.7, cy + 1.2), arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#d97706"))
    ax.add_patch(FancyArrowPatch((cx, cy + 0.6), (xs[3] + w / 2, y), connectionstyle="arc3,rad=-0.25",
                                 arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#d97706"))
    ax.text(xs[3] - 0.6, 0.85, "next iteration", ha="center", va="center", fontsize=12, color="#b45309", style="italic")
    save(fig, "fig1_architecture")


# ---------------- Figure 2: task scenes rendered in PyBullet ----------------
def fig2():
    import pybullet as p
    from simulation.env import RoboTeachEnv
    from calibration.ontology import CanonicalTaskOntology
    from benchmark.ablation_suite import BENCHMARK_TASKS

    names = ["Tower", "Row", "Diagonal", "Box", "Swap", "Obstacle"]
    env = RoboTeachEnv(headless=True, seed=500, include_container=True, include_obstacle=False)
    ont = CanonicalTaskOntology(llm_client=None)
    imgs = []
    view = p.computeViewMatrix(cameraEyePosition=[0.72, -0.62, 0.95], cameraTargetPosition=[0.12, 0.02, 0.43],
                               cameraUpVector=[0, 0, 1])
    proj = p.computeProjectionMatrixFOV(fov=48, aspect=4 / 3, nearVal=0.05, farVal=4.0)
    for t_idx, item in enumerate(BENCHMARK_TASKS):
        env.reset()
        env.randomize(ont.parse(item["command_tr"]).to_scene_config(), seed=3000 + t_idx * 100)
        for _ in range(120):
            p.stepSimulation(physicsClientId=env.client)
        _, _, rgb, _, _ = p.getCameraImage(1200, 900, view, proj, renderer=p.ER_TINY_RENDERER,
                                           shadow=1, lightDirection=[0.4, -0.6, 1.0], physicsClientId=env.client)
        imgs.append(np.reshape(rgb, (900, 1200, 4))[:, :, :3])
    env.close()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    for ax, img, name in zip(axes.ravel(), imgs, names):
        ax.imshow(img)
        ax.set_title(name, fontsize=15, weight="bold", color=INK, pad=6)
        ax.axis("off")
    fig.tight_layout(pad=0.6)
    save(fig, "fig2_tasks")


# ---------------- Figure 3: camera layout and occluded cube ----------------
def fig3():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.05]})
    # (a) top view of the workspace
    a.add_patch(Rectangle((-0.40, -0.55), 1.10, 1.10, fc="#f3f4f6", ec=GREY, lw=1.2))
    a.add_patch(Rectangle((-0.48, -0.08), 0.16, 0.16, fc="#d1d5db", ec=INK))
    a.text(-0.40, -0.13, "robot", ha="center", va="top", fontsize=12, color=INK)
    a.add_patch(Rectangle((0.02, -0.22), 0.28, 0.40, fc="none", ec=GREY, lw=1, ls="--"))
    a.text(0.16, -0.27, "work area", ha="center", va="top", fontsize=11, color=GREY)
    cams = {"top (above)": (0.16, 0.0), "front": (0.68, -0.48), "side": (0.16, 0.78), "opposite": (0.68, 0.48)}
    offs = {"top (above)": (0.0, 0.08), "front": (0.0, -0.09), "side": (0.0, 0.08), "opposite": (0.13, 0.07)}
    for name, (x, y) in cams.items():
        a.add_patch(Circle((x, y), 0.035, fc="white", ec=BLUE, lw=2))
        dx, dy = offs[name]
        a.text(x + dx, y + dy, name, ha="center", va="center", fontsize=12, color=BLUE, weight="bold")
    a.set_xlim(-0.6, 0.95)
    a.set_ylim(-0.7, 0.9)
    a.set_aspect("equal")
    a.axis("off")
    a.set_title("(a) Camera layout (top view)", fontsize=14, color=INK)
    # (b) side view: hidden cube under another cube
    b.plot([-0.2, 1.3], [0, 0], color=GREY, lw=2)
    b.add_patch(Rectangle((0.3, 0.0), 0.3, 0.3, fc="#2563eb", ec=INK, lw=1.2))
    b.add_patch(Rectangle((0.3, 0.3), 0.3, 0.3, fc="#dc2626", ec=INK, lw=1.2))
    b.text(0.25, 0.15, "hidden\ncube", ha="right", va="center", color=INK, fontsize=12)
    b.add_patch(Circle((0.45, 1.05), 0.05, fc="white", ec=BLUE, lw=2))
    b.text(0.45, 1.17, "top camera", ha="center", fontsize=12, color=BLUE)
    b.annotate("", xy=(0.45, 0.62), xytext=(0.45, 1.0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4))
    b.text(0.62, 0.8, "sees only\nthe upper cube", fontsize=11, color=GREY, va="center")
    b.add_patch(Circle((1.2, 0.35), 0.05, fc="white", ec=BLUE, lw=2))
    b.text(1.2, 0.48, "side camera", ha="center", fontsize=12, color=BLUE)
    b.annotate("", xy=(0.61, 0.15), xytext=(1.15, 0.33), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4))
    b.plot([0.6], [0.15], "o", color=INK, ms=6)
    b.text(0.66, 0.05, "visible face", fontsize=11, color=INK)
    b.plot([0.45], [0.15], "x", color="white", ms=10, mew=2.5)
    b.annotate("", xy=(0.47, 0.15), xytext=(0.6, 0.15), arrowprops=dict(arrowstyle="-|>", color="#facc15", lw=2))
    b.text(0.18, -0.12, "estimated center = visible face + half width", fontsize=11, color=INK)
    b.set_xlim(-0.25, 1.4)
    b.set_ylim(-0.25, 1.3)
    b.set_aspect("equal")
    b.axis("off")
    b.set_title("(b) Locating a hidden cube", fontsize=14, color=INK)
    fig.tight_layout()
    save(fig, "fig3_cameras")


# ---------------- Figure 4: main results ----------------
def fig4():
    res = json.load(open(os.path.join(ROOT, "benchmark", "results", "live_ablation_benchmark.json")))["results_by_method"]
    keys = ["zero_shot_open_loop", "fair_cap_multiview", "unstructured_verbal_reflexion", "neuro_symbolic_roboteach"]
    labels = ["Single camera", "Four cameras", "Verbal correction", "RoboTeach"]
    colors = ["#d1d5db", "#93c5fd", "#fcd34d", "#2563eb"]
    tasks = ["T1_STACK_ORDERED", "T2_ALIGN_ROW", "T3_DIAGONAL_PAIR", "T4_PLACE_INSIDE", "T5_UNSTACK_SWAP", "T6_OBSTACLE_CLEARANCE"]
    names = ["Tower", "Row", "Diagonal", "Box", "Swap", "Obstacle"]
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(tasks))
    bw = 0.2
    for i, (k, lab, col) in enumerate(zip(keys, labels, colors)):
        vals = [100.0 * res[k][t]["success"] / res[k][t]["trials"] for t in tasks]
        ax.bar(x + (i - 1.5) * bw, vals, bw, label=lab, color=col, edgecolor=INK, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=4, frameon=False, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    save(fig, "fig4_results")


if __name__ == "__main__":
    fig1()
    fig3()
    fig4()
    fig2()
    print("done", sorted(os.listdir(OUT)))
