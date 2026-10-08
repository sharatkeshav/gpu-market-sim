"""Summaries of results/*.json (mean and 95% CI across seeds)."""
import json, math, sys, os
from collections import defaultdict
import numpy as np
from scipy import stats
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def load(name):
    return json.load(open(os.path.join(R, f"{name}.json")))

def ci(xs):
    """Mean and half-width of a two-sided 95% Student-t confidence interval."""
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return float(xs.mean()), float("nan")
    t = float(stats.t.ppf(0.975, len(xs) - 1))
    return float(xs.mean()), float(t * xs.std(ddof=1) / math.sqrt(len(xs)))

def group(res, keys, metric):
    g = defaultdict(list)
    for m in res:
        g[tuple(m["_kw"].get(k) for k in keys)].append(metric(m))
    return {k: ci(v) for k, v in g.items()}

if __name__ == "__main__":
    e1 = load("e1_policies")
    for name, f in [("cost", lambda m: m["cost_per_useful_gpu_h"]),
                    ("train$", lambda m: m["by_class"]["train"]["cost"]),
                    ("small$", lambda m: m["by_class"]["small"]["cost"]),
                    ("jct", lambda m: m["jct_mean"]), ("p95", lambda m: m["jct_p95"]),
                    ("wait", lambda m: m["wait_mean"]), ("ovh", lambda m: m["interrupt_overhead_mean"]),
                    ("util", lambda m: m["utilization"]), ("comm_share", lambda m: m["tier_share"]["community"])]:
        print("==", name)
        g = group(e1, ["load", "policy"], f)
        for k in sorted(g):
            print(f"  load={k[0]:<5} {k[1]:18s} {g[k][0]:8.3f} ± {g[k][1]:.3f}")
