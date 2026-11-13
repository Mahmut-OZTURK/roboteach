# stats.py — Exact Binomial/Wilson CIs, Paired McNemar Tests & Bootstrap Percentiles (stdlib only)
import math
import random


def wilson_ci(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% confidence interval in percentage [0.0, 100.0]."""
    if total <= 0:
        return (0.0, 0.0)
    p = max(0.0, min(1.0, successes / total))
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)) / denom
    lo = max(0.0, (center - half) * 100.0)
    hi = min(100.0, (center + half) * 100.0)
    return (round(lo, 2), round(hi, 2))


def mcnemar_exact(outcomes_a: list[bool], outcomes_b: list[bool]) -> dict:
    """
    Two-sided exact McNemar test on matched binary trial outcomes.
    b: A=True, B=False; c: A=False, B=True.
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("Matched outcome arrays must have equal length")
    b = sum(1 for a, b_v in zip(outcomes_a, outcomes_b) if bool(a) and not bool(b_v))
    c = sum(1 for a, b_v in zip(outcomes_a, outcomes_b) if (not bool(a)) and bool(b_v))
    n_disc = b + c
    if n_disc == 0:
        return {"n_matched": len(outcomes_a), "b_only_a": 0, "c_only_b": 0, "p_value": 1.0}
    k_max = min(b, c)
    prob_tail = sum(math.comb(n_disc, k) * (0.5 ** n_disc) for k in range(k_max + 1))
    p_val = min(1.0, 2.0 * prob_tail)
    return {
        "n_matched": len(outcomes_a),
        "b_only_a": b,
        "c_only_b": c,
        "p_value": round(p_val, 6),
    }


def _quantile(sorted_vals: list[float], q: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    idx = q * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def percentile_summary(values_ms: list[float], n_boot: int = 1000, seed: int = 42) -> dict:
    """Computes mean, p50, p95, p99 and 95% bootstrap CIs for p50 and p95 in milliseconds."""
    if not values_ms:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "p50_ci95": [0.0, 0.0], "p95_ci95": [0.0, 0.0]}
    sv = sorted(float(v) for v in values_ms)
    n = len(sv)
    rng = random.Random(seed)
    boot_p50, boot_p95 = [], []
    for _ in range(n_boot):
        sample = sorted(sv[rng.randrange(n)] for _ in range(n))
        boot_p50.append(_quantile(sample, 0.50))
        boot_p95.append(_quantile(sample, 0.95))
    boot_p50.sort()
    boot_p95.sort()
    return {
        "n": n,
        "mean": round(sum(sv) / n, 2),
        "p50": round(_quantile(sv, 0.50), 2),
        "p95": round(_quantile(sv, 0.95), 2),
        "p99": round(_quantile(sv, 0.99), 2),
        "p50_ci95": [round(_quantile(boot_p50, 0.025), 2), round(_quantile(boot_p50, 0.975), 2)],
        "p95_ci95": [round(_quantile(boot_p95, 0.025), 2), round(_quantile(boot_p95, 0.975), 2)],
    }

