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
    b: A=True, B=False; c: A=False, B=True. Preserves full float precision on p_value.
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
        "p_value": float(p_val),
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


def task_stratified_paired_bootstrap_ci(
    per_task_a: dict[str, list[bool]],
    per_task_b: dict[str, list[bool]],
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Task-stratified paired bootstrap 95% CI for difference in success rate (A - B, in pp).
    Resamples matched seed indices with replacement within each task stratum.
    """
    tasks = list(per_task_a.keys())
    if set(tasks) != set(per_task_b.keys()) or not tasks:
        raise ValueError("Task strata mismatch or empty task mapping")
    paired_deltas: list[list[int]] = []
    tot_a, tot_b, total_n = 0, 0, 0
    for t in tasks:
        oa, ob = per_task_a[t], per_task_b[t]
        if len(oa) != len(ob) or not oa:
            raise ValueError(f"Task {t} matched outcome length mismatch or empty")
        deltas = [int(bool(a)) - int(bool(b)) for a, b in zip(oa, ob)]
        paired_deltas.append(deltas)
        tot_a += sum(bool(a) for a in oa)
        tot_b += sum(bool(b) for b in ob)
        total_n += len(oa)

    obs_diff_pp = 100.0 * (tot_a - tot_b) / total_n
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        acc = 0
        for deltas in paired_deltas:
            acc += sum(rng.choices(deltas, k=len(deltas)))
        boots.append(100.0 * acc / total_n)
    boots.sort()
    return {
        "n_matched": total_n,
        "n_strata": len(tasks),
        "success_a": tot_a,
        "success_b": tot_b,
        "rate_a_pct": round(100.0 * tot_a / total_n, 2),
        "rate_b_pct": round(100.0 * tot_b / total_n, 2),
        "diff_pp": round(obs_diff_pp, 2),
        "ci95_pp": [round(_quantile(boots, 0.025), 2), round(_quantile(boots, 0.975), 2)],
        "n_boot": n_boot,
        "seed": seed,
    }


def _compute_cluster_metric(vals: list[float], metric: str, threshold: float | None = None) -> float:
    n = len(vals)
    if n == 0:
        return 0.0
    if metric == "rmse":
        return math.sqrt(sum(v * v for v in vals) / n)
    if metric == "mae":
        return sum(abs(v) for v in vals) / n
    if metric == "mean":
        return sum(vals) / n
    if metric == "median":
        return _quantile(sorted(vals), 0.50)
    if metric == "iqr":
        sv = sorted(vals)
        return _quantile(sv, 0.75) - _quantile(sv, 0.25)
    if metric == "rate_lt":
        if threshold is None:
            raise ValueError("threshold is required for rate_lt metric")
        return 100.0 * sum(1 for v in vals if v < threshold) / n
    raise ValueError(f"Unsupported cluster metric: {metric}")


def scene_clustered_bootstrap_ci(
    scene_clusters: list[list[float]] | dict[str, list[float]],
    metric: str = "mean",
    threshold: float | None = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Scene-clustered bootstrap 95% CI resampling independent scene clusters with replacement
    and pooling all queries within each resampled cluster.
    """
    clusters = list(scene_clusters.values()) if isinstance(scene_clusters, dict) else list(scene_clusters)
    clusters = [[float(x) for x in c] for c in clusters if len(c) > 0]
    n_clusters = len(clusters)
    pooled = [x for c in clusters for x in c]
    if n_clusters == 0 or not pooled:
        return {"n_clusters": 0, "n_obs": 0, "estimate": 0.0, "ci95": [0.0, 0.0], "n_boot": n_boot, "seed": seed}

    est = _compute_cluster_metric(pooled, metric, threshold=threshold)
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        sample_vals: list[float] = []
        for cl in rng.choices(clusters, k=n_clusters):
            sample_vals.extend(cl)
        boots.append(_compute_cluster_metric(sample_vals, metric, threshold=threshold))
    boots.sort()
    return {
        "n_clusters": n_clusters,
        "n_obs": len(pooled),
        "estimate": round(est, 2),
        "ci95": [round(_quantile(boots, 0.025), 2), round(_quantile(boots, 0.975), 2)],
        "n_boot": n_boot,
        "seed": seed,
    }
