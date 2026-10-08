"""Writes LaTeX tables and a numbers file (paper/generated/*.tex) from results."""
import json
import math
import os
from collections import defaultdict

import numpy as np

from analyze import ci, group, load

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "generated")
os.makedirs(OUT, exist_ok=True)
macros = {}


def write_tab(fname, cols, header, rows):
    body = "\n".join(rows)
    tex = (f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
           f"{body}\n\\bottomrule\n\\end{{tabular}}\n")
    open(os.path.join(OUT, fname), "w").write(tex)


def mac(name, val):
    macros[name] = val


# --- validation -------------------------------------------------------------
val = load("validation")
rel = [abs(r["sim_mean"] - r["exact"]) / r["exact"] for r in val]
inci = sum(abs(r["sim_mean"] - r["exact"]) <= r["sim_ci"] for r in val)
daly_rel = [abs(r["daly"] - r["exact"]) / r["exact"] for r in val]
mac("ValMaxRelErr", f"{100 * max(rel):.2f}")
mac("ValInCI", f"{inci}")
mac("ValPoints", f"{len(val)}")
mac("DalyMaxRelErr", f"{100 * max(daly_rel):.1f}")
calib = {(r["mtbf"], r["tau"]): r for r in val}
mac("CalibCommOvh", f"{100 * calib[(4.5, 1.0)]['exact_overhead']:.0f}")
mac("CalibVerOvh", f"{100 * calib[(16, 1.0)]['exact_overhead']:.1f}")
mac("CalibCommSim", f"{100 * calib[(4.5, 1.0)]['sim_overhead']:.1f}")
mac("CalibVerSim", f"{100 * calib[(16, 1.0)]['sim_overhead']:.1f}")

rows = []
for r in sorted(val, key=lambda r: (r["mtbf"], r["tau"])):
    if r["tau"] != 1.0:
        continue
    rows.append(f"{r['mtbf']:g} & {r['exact']:.2f} & {r['sim_mean']:.2f} $\\pm$ {r['sim_ci']:.2f} & "
                f"{100 * r['exact_overhead']:.1f} & {100 * r['sim_overhead']:.1f} \\\\")
write_tab("tab_validation.tex", "rrrrr", "MTBF (h) & Exact & Simulated & Ovh.\\ exact & Ovh.\\ sim.", rows)

# --- E1 paired --------------------------------------------------------------
e1 = load("e1_policies")
d = defaultdict(dict)
for m in e1:
    d[(m["_kw"]["load"], m["_kw"]["seed"])][m["_kw"]["policy"]] = m


def pdiff(L, p, base="cheapest", f=lambda m: m["cost_per_useful_gpu_h"]):
    xs = [100 * (f(d[(L, s)][p]) / f(d[(L, s)][base]) - 1) for s in range(8)
          if p in d[(L, s)] and base in d[(L, s)]]
    return ci(xs)


def fmt(x):
    return f"{x:.1f}"


for L, tag in [(0.2, "Lo"), (0.35, "Mid"), (0.5, "Half"), (0.8, "Hi")]:
    for p, ptag in [("rem", "Rem"), ("rem_oracle", "Oracle"), ("threshold", "Thr"),
                    ("verified_cheapest", "Ver"), ("most_reliable", "Rel"), ("random", "Rand")]:
        mu, h = pdiff(L, p)
        mac(f"E{ptag}{tag}", f"{abs(mu):.1f}")  # direction stated in prose
        mac(f"E{ptag}{tag}CI", f"{h:.1f}")

g_wait = group(e1, ["load", "policy"], lambda m: m["wait_mean"])
g_p95 = group(e1, ["load", "policy"], lambda m: m["jct_p95"])
g_jct = group(e1, ["load", "policy"], lambda m: m["jct_mean"])
mac("WaitVerHalf", f"{g_wait[(0.5, 'verified_cheapest')][0]:.1f}")
mac("WaitVerHi", f"{g_wait[(0.8, 'verified_cheapest')][0]:.1f}")
mac("WaitRemHi", f"{g_wait[(0.8, 'rem')][0]:.1f}")
mac("WaitThrMidHi", f"{g_wait[(0.65, 'threshold')][0]:.1f}")
mac("WaitRemMidHi", f"{g_wait[(0.65, 'rem')][0]:.2f}")
mac("PNinetyFiveThrHi", f"{g_p95[(0.8, 'threshold')][0]:.1f}")
mac("PNinetyFiveRemHi", f"{g_p95[(0.8, 'rem')][0]:.1f}")
mac("JctRemLo", f"{g_jct[(0.2, 'rem')][0]:.2f}")
mac("JctThrLo", f"{g_jct[(0.2, 'threshold')][0]:.2f}")
mac("JctCheapLo", f"{g_jct[(0.2, 'cheapest')][0]:.2f}")
mac("JctRelLo", f"{g_jct[(0.2, 'most_reliable')][0]:.2f}")
g_cost = group(e1, ["load", "policy"], lambda m: m["cost_per_useful_gpu_h"])
mac("CostRemLo", f"{g_cost[(0.2, 'rem')][0]:.2f}")
mac("CostCheapLo", f"{g_cost[(0.2, 'cheapest')][0]:.2f}")
g_share = group(e1, ["load", "policy"], lambda m: m["tier_share"]["community"])
mac("ShareRemLo", f"{100 * g_share[(0.2, 'rem')][0]:.0f}")
mac("ShareCheapLo", f"{100 * g_share[(0.2, 'cheapest')][0]:.0f}")
mac("CommDiscount", f"{100 * (1 - 1.10 / ((1.50 + 1.87) / 2)):.0f}")
mac("ShareThrLo", f"{100 * g_share[(0.2, 'threshold')][0]:.0f}")
mu, h = pdiff(0.2, "rem", f=lambda m: m["by_class"]["train"]["cost"])
mac("ERemTrainLo", f"{abs(mu):.1f}")
mu, h = pdiff(0.2, "rem", f=lambda m: m["by_class"]["small"]["cost"])
mac("ERemSmallLo", f"{abs(mu):.1f}")
mu, h = pdiff(0.2, "rem", "threshold")
mac("ERemVsThrLo", f"{abs(mu):.1f}")

# E1 main table: cost and JCT at load 0.35
POLS = [("random", "Random"), ("cheapest", "Cheapest"), ("most_reliable", "Most-reliable"),
        ("verified_cheapest", "Verified-only"), ("threshold", "Threshold (4\\,h)"),
        ("rem", "\\textbf{REM (ours)}"), ("rem_oracle", "REM, oracle $\\lambda$")]
rows = []
for p, name in POLS:
    c = g_cost[(0.35, p)]
    dm, dh = pdiff(0.35, p) if p != "cheapest" else (0.0, 0.0)
    j = g_jct[(0.35, p)]
    o = group([m for m in e1 if m["_kw"]["load"] == 0.35 and m["_kw"]["policy"] == p],
              ["policy"], lambda m: m["interrupt_overhead_mean"])[(p,)]
    sh = g_share[(0.35, p)][0]
    dstr = "---" if p == "cheapest" else f"{dm:+.1f} $\\pm$ {dh:.1f}".replace("-", "$-$", 1)
    rows.append(f"{name} & {c[0]:.3f} & {dstr} & {100 * o[0]:.1f} & {j[0]:.2f} & {100 * sh:.0f} \\\\")
write_tab("tab_policies.tex", "lrrrrr", "Policy & Cost (\\$) &$\\Delta$ (\\%) & Ovh.\\ (\\%) & JCT (h) & Comm.\\ (\\%)", rows)

# --- E2 ---------------------------------------------------------------------
e2 = load("e2_sensitivity")
d2 = defaultdict(dict)
for m in e2:
    t = m["_kw"]["tiers"][0]
    d2[(t["mtbf_median"], round((t["price_lo"] + t["price_hi"]) / 2, 2), m["_kw"]["seed"])][m["_kw"]["policy"]] = m["cost_per_useful_gpu_h"]
cells = defaultdict(list)
for (mt, pm, s), v in d2.items():
    cells[(mt, pm)].append(v)
rc = {k: np.mean([100 * (1 - x["rem"] / x["cheapest"]) for x in v]) for k, v in cells.items()}
rt = {k: np.mean([100 * (1 - x["rem"] / x["threshold"]) for x in v]) for k, v in cells.items()}
tc = {k: np.mean([100 * (x["threshold"] / x["cheapest"] - 1) for x in v]) for k, v in cells.items()}
mac("SensRemMax", f"{max(rc.values()):.1f}")
mac("SensRemMin", f"{min(rc.values()):.1f}")
mac("SensRemThrMax", f"{max(rt.values()):.1f}")
mac("SensRemThrMin", f"{min(rt.values()):.1f}")
mac("SensThrWorst", f"{max(tc.values()):.0f}")
mac("SensThrBest", f"{min(tc.values()):.1f}")
mac("SensThrBeatsCount", f"{sum(v < 0 for v in tc.values())}")
mac("SensCells", f"{len(tc)}")
worst_rem_seed = min(100 * (1 - x["rem"] / x["cheapest"]) for v in cells.values() for x in v)
mac("SensRemWorstSeed", f"{abs(min(worst_rem_seed, 0.0)):.1f}")

# --- E3 ---------------------------------------------------------------------
e3 = load("e3_checkpoint")
rows = []
for ck, tau, name in [("fixed", 0.5, "Fixed, $\\tau=0.5$\\,h"), ("fixed", 1.0, "Fixed, $\\tau=1$\\,h"),
                      ("fixed", 2.0, "Fixed, $\\tau=2$\\,h"), ("daly", None, "Per-host Young/Daly")]:
    cells = []
    for p in ["cheapest", "rem"]:
        sel = [m for m in e3 if m["_kw"]["ckpt"] == ck and m["_kw"].get("fixed_tau") == tau and m["_kw"]["policy"] == p]
        c = ci([m["cost_per_useful_gpu_h"] for m in sel])
        o = ci([m["tier_overhead"]["community"] for m in sel])
        cells += [f"{c[0]:.3f}", f"{100 * o[0]:.1f}"]
    rows.append(name + " & " + " & ".join(cells) + " \\\\")
write_tab("tab_checkpoint.tex", "lrrrr",
          " & \\multicolumn{2}{c}{Cheapest} & \\multicolumn{2}{c}{REM} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nCheckpoint interval & \\$/GPU-h & Ovh.\\ (\\%) & \\$/GPU-h & Ovh.\\ (\\%)", rows)


def e3cost(ck, tau, p):
    return np.mean([m["cost_per_useful_gpu_h"] for m in e3 if m["_kw"]["ckpt"] == ck
                    and m["_kw"].get("fixed_tau") == tau and m["_kw"]["policy"] == p])


mac("CkOneOverDaly", f"{100 * (e3cost('fixed', 1.0, 'cheapest') / e3cost('daly', None, 'cheapest') - 1):.1f}")
mac("CkTwoOverDaly", f"{100 * (e3cost('fixed', 2.0, 'cheapest') / e3cost('daly', None, 'cheapest') - 1):.1f}")
mac("CkRemSaveTwo", f"{100 * (1 - e3cost('fixed', 2.0, 'rem') / e3cost('fixed', 2.0, 'cheapest')):.1f}")
mac("CkCommOvhTwo", f"{100 * np.mean([m['tier_overhead']['community'] for m in e3 if m['_kw']['ckpt'] == 'fixed' and m['_kw'].get('fixed_tau') == 2.0 and m['_kw']['policy'] == 'cheapest']):.0f}")
mac("CkCommOvhDaly", f"{100 * np.mean([m['tier_overhead']['community'] for m in e3 if m['_kw']['ckpt'] == 'daly' and m['_kw']['policy'] == 'cheapest']):.1f}")

# --- E4 ---------------------------------------------------------------------
e4 = load("e4_mig")
sp = group(e4, ["load", "mig"], lambda m: m["total_spend"])
sw = group(e4, ["load", "mig"], lambda m: m["by_class"]["small"]["wait"])
jc = group(e4, ["load", "mig"], lambda m: m["jct_mean"])
savings = [100 * (1 - sp[(L, True)][0] / sp[(L, False)][0]) for L in [0.2, 0.35, 0.5, 0.65, 0.8]]
mac("MigSaveMin", f"{min(savings):.0f}")
mac("MigSaveMax", f"{max(savings):.0f}")
mac("MigSmallWaitWholeHi", f"{sw[(0.8, False)][0]:.1f}")
mac("MigSmallWaitMigHi", f"{sw[(0.8, True)][0]:.1f}")
mac("MigSmallWaitWholeMH", f"{sw[(0.65, False)][0]:.1f}")
mac("MigSmallWaitMigMH", f"{60 * sw[(0.65, True)][0]:.1f}")
mac("MigJctWholeHi", f"{jc[(0.8, False)][0]:.1f}")
mac("MigJctMigHi", f"{jc[(0.8, True)][0]:.1f}")
mac("MigJctWholeMH", f"{jc[(0.65, False)][0]:.1f}")
mac("MigJctMigMH", f"{jc[(0.65, True)][0]:.1f}")

# --- E5 ---------------------------------------------------------------------
e5 = load("e5_reputation")
K5 = ["reputation", "ckpt_reputation", "history_hours"]
g5 = group(e5, K5, lambda m: m["cost_per_useful_gpu_h"])
gt = group(e5, K5, lambda m: m["jct_mean"])
orc = g5[("oracle", None, 0.0)][0]
rows = []
for key, name in [(("tier", None, 0.0), "Tier average, both uses"),
                  (("tier", "learned", 72.0), "Tier average, placement only"),
                  (("learned", None, 0.0), "Per-host, no history"),
                  (("learned", None, 24.0), "Per-host, 24\\,h history"),
                  (("learned", None, 72.0), "Per-host, 72\\,h history"),
                  (("learned", None, 336.0), "Per-host, 2-week history"),
                  (("oracle", None, 0.0), "Oracle (true $\\lambda$)")]:
    c = g5[key]
    rows.append(f"{name} & {c[0]:.3f} $\\pm$ {c[1]:.3f} & {100 * (c[0] / orc - 1):+.1f} & {gt[key][0]:.2f} \\\\")
write_tab("tab_reputation.tex", "lrrr", "Failure-rate estimate & Cost (\\$) & Gap (\\%) & JCT (h)", rows)
mac("RepTierGap", f"{100 * (g5[('tier', None, 0.0)][0] / orc - 1):.1f}")
mac("RepTierPlaceGap", f"{100 * (g5[('tier', 'learned', 72.0)][0] / orc - 1):.1f}")
mac("RepColdGap", f"{100 * (g5[('learned', None, 0.0)][0] / orc - 1):.1f}")
mac("RepWeekGap", f"{100 * (g5[('learned', None, 336.0)][0] / orc - 1):.1f}")

with open(os.path.join(OUT, "numbers.tex"), "w") as f:
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(json.dumps(macros, indent=0))

# tier-only REM vs Cheapest at the same load and seeds (E5 vs E1)
with open(os.path.join(OUT, "numbers.tex"), "a") as f:
    f.write(f"\\newcommand{{\\RepTierCost}}{{{g5[('tier', None, 0.0)][0]:.3f}}}\n")
    f.write(f"\\newcommand{{\\RepTierPlaceCost}}{{{g5[('tier', 'learned', 72.0)][0]:.3f}}}\n")
    f.write(f"\\newcommand{{\\CheapMidCost}}{{{g_cost[(0.35, 'cheapest')][0]:.3f}}}\n")


# ===========================================================================
# Revision experiments E6-E10
# ===========================================================================
rev = {}


def rmac(name, val):
    rev[name] = val


def rel_by(res, keyf, polf, metric, base="cheapest"):
    d = defaultdict(dict)
    for m in res:
        d[keyf(m)][polf(m)] = m
    out = defaultdict(list)
    for (grp, seed), v in d.items():
        for p, m in v.items():
            if p != base and base in v:
                out[(grp, p)].append(100 * (metric(m) / metric(v[base]) - 1))
    return {k: ci(v) for k, v in out.items()}


cost = lambda m: m["cost_per_useful_gpu_h"]

# --- E6 sigma ---
# variants: placement signal | checkpoint signal, compared with Cheapest
# (per-host checkpoints) at the same spread and seed
e6 = load("e6_sigma")


def lab6(m):
    kw = m["_kw"]
    rep = kw.get("reputation", "learned")
    ck = kw.get("ckpt_reputation") or rep
    return {("cheapest", "learned", "learned"): "cheapest",
            ("cheapest", "tier", "tier"): "cheap_tierckpt",
            ("rem", "tier", "learned"): "rem_tierplace",
            ("rem", "tier", "tier"): "rem_tier",
            ("rem", "learned", "learned"): "rem",
            ("rem_oracle", "learned", "learned"): "rem_oracle",
            ("threshold", "learned", "learned"): "threshold"}[(kw["policy"], rep, ck)]


g6 = rel_by(e6, lambda m: (m["_kw"]["tiers"][0]["mtbf_sigma"], m["_kw"]["seed"]), lab6, cost)
S6 = [0.0, 0.4, 0.8, 1.2]
rmac("SigRemZero", f"{abs(g6[(0.0, 'rem')][0]):.1f}")
rmac("SigRemZeroCI", f"{g6[(0.0, 'rem')][1]:.1f}")
rmac("SigOracleZero", f"{abs(g6[(0.0, 'rem_oracle')][0]):.1f}")
rmac("SigRemFour", f"{abs(g6[(0.4, 'rem')][0]):.1f}")
rmac("SigRemBest", f"{abs(min(g6[(s, 'rem')][0] for s in S6)):.1f}")
rmac("SigPlaceZero", f"{abs(g6[(0.0, 'rem_tierplace')][0]):.1f}")
rmac("SigPlaceMax", f"{max(abs(g6[(s, 'rem_tierplace')][0]) for s in S6):.1f}")
rmac("SigPlaceEight", f"{g6[(0.8, 'rem_tierplace')][0]:+.1f}".replace("-", "$-$"))
rmac("SigPlaceTwelve", f"{g6[(1.2, 'rem_tierplace')][0]:+.1f}".replace("-", "$-$"))
rmac("SigPlaceTwelveCI", f"{g6[(1.2, 'rem_tierplace')][1]:.1f}")
rmac("SigCkptEight", f"{g6[(0.8, 'cheap_tierckpt')][0]:.1f}")
rmac("SigCkptTwelve", f"{g6[(1.2, 'cheap_tierckpt')][0]:.1f}")
rmac("SigBothTwelve", f"{g6[(1.2, 'rem_tier')][0]:.1f}")
rmac("SigThrLo", f"{g6[(0.0, 'threshold')][0]:.0f}")
rmac("SigThrHi", f"{g6[(1.2, 'threshold')][0]:.0f}")

# --- E7 failure model ---
e7 = load("e7_failmodel")
g7 = rel_by(e7, lambda m: ((m["_kw"]["weibull_k"], m["_kw"]["corr_frac"]), m["_kw"]["seed"]),
            lambda m: m["_kw"]["policy"], cost)
rows = []
for k in [0.7, 1.0, 1.5]:
    for c in [0.0, 0.5]:
        sel = [m for m in e7 if m["_kw"]["weibull_k"] == k and m["_kw"]["corr_frac"] == c
               and m["_kw"]["policy"] == "cheapest"]
        cc = ci([m["cost_per_useful_gpu_h"] for m in sel])[0]
        ov = np.mean([m["tier_overhead"]["community"] for m in sel])
        r, t = g7[((k, c), "rem")], g7[((k, c), "threshold")]
        kname = {0.7: "0.7", 1.0: "1.0", 1.5: "1.5"}[k]
        rows.append(f"{kname} & {int(c * 100)}\\% & {cc:.3f} & {100 * ov:.1f} & "
                    f"$-${abs(r[0]):.1f} $\\pm$ {r[1]:.1f} & +{t[0]:.1f} $\\pm$ {t[1]:.1f} \\\\")
write_tab("tab_failmodel.tex", "lrrrrr",
          "$k$ & Corr. & Cheapest (\\$) & Comm.\\ ovh.\\ (\\%) & REM $\\Delta$ (\\%) & Thr.\\ $\\Delta$ (\\%)", rows)
rem7 = [g7[((k, c), "rem")][0] for k in [0.7, 1.0, 1.5] for c in [0.0, 0.5]]
thr7 = [g7[((k, c), "threshold")][0] for k in [0.7, 1.0, 1.5] for c in [0.0, 0.5]]
rmac("RobRemMin", f"{abs(max(rem7)):.1f}")
rmac("RobRemMax", f"{abs(min(rem7)):.1f}")
rmac("RobThrMin", f"{min(thr7):.0f}")
rmac("RobThrMax", f"{max(thr7):.0f}")

# --- E8 gang ---
e8 = load("e8_gang")
key8 = lambda m: ((m["_kw"]["load"], m["_kw"]["corr_frac"]), m["_kw"]["seed"])
pol = lambda m: m["_kw"]["policy"]
mn = lambda m: m["by_class"]["multinode"]["cost"]
g8all = rel_by(e8, key8, pol, cost)
g8mn = rel_by(e8, key8, pol, mn)
g8vsrel = rel_by(e8, key8, pol, mn, base="most_reliable")
grid8 = [(L, c) for L in [0.2, 0.35, 0.5] for c in [0.0, 0.5]]
remmn = [g8mn[(g, "rem")][0] for g in grid8]
remall = [g8all[(g, "rem")][0] for g in grid8]
thrmn = [g8mn[(g, "threshold")][0] for g in grid8]
relmn = [g8mn[(g, "most_reliable")][0] for g in grid8]
remvrel = [g8vsrel[(g, "rem")][0] for g in grid8]
rmac("GangRemMnMin", f"{abs(max(remmn)):.0f}")
rmac("GangRemMnMax", f"{abs(min(remmn)):.0f}")
rmac("GangRemAllMin", f"{abs(max(remall)):.0f}")
rmac("GangRemAllMax", f"{abs(min(remall)):.0f}")
rmac("GangThrMnMin", f"{abs(max(thrmn)):.0f}")
rmac("GangThrMnMax", f"{abs(min(thrmn)):.0f}")
rmac("GangRelMnMin", f"{abs(max(relmn)):.0f}")
rmac("GangRelMnMax", f"{abs(min(relmn)):.0f}")
rmac("GangRemVsRelMin", f"{abs(max(remvrel)):.0f}")
rmac("GangRemVsRelMax", f"{abs(min(remvrel)):.0f}")


def g8m(L, c, p, f):
    return np.mean([f(m) for m in e8 if m["_kw"]["load"] == L and m["_kw"]["corr_frac"] == c
                    and m["_kw"]["policy"] == p])


for p, tag in [("cheapest", "Cheap"), ("rem", "Rem"), ("most_reliable", "Rel"), ("threshold", "Thr")]:
    rmac(f"GangJct{tag}", f"{g8m(0.35, 0.0, p, lambda m: m['by_class']['multinode']['jct']):.1f}")
    rmac(f"GangOvh{tag}", f"{100 * g8m(0.35, 0.0, p, lambda m: m['by_class']['multinode']['overhead']):.0f}")
    rmac(f"GangWaitHalf{tag}", f"{g8m(0.5, 0.0, p, lambda m: m['by_class']['multinode']['wait']):.1f}")
rmac("GangJctRelLo", f"{g8m(0.2, 0.0, 'most_reliable', lambda m: m['by_class']['multinode']['jct']):.1f}")
rmac("GangJctRemLo", f"{g8m(0.2, 0.0, 'rem', lambda m: m['by_class']['multinode']['jct']):.1f}")
rmac("GangShare", f"{100 * np.mean([m['by_class']['multinode']['n'] / m['n_done'] for m in e8]):.0f}")

# --- E9 scale ---
e9 = load("e9_scale")
g9 = rel_by(e9, lambda m: (m["_kw"]["load"], m["_kw"]["seed"]), pol, cost)
rmac("ScaleRemMid", f"{abs(g9[(0.35, 'rem')][0]):.1f}")
rmac("ScaleRemMidCI", f"{g9[(0.35, 'rem')][1]:.1f}")
rmac("ScaleRemHalf", f"{abs(g9[(0.5, 'rem')][0]):.1f}")
rmac("ScaleThrMid", f"{g9[(0.35, 'threshold')][0]:.1f}")
rmac("ScaleThrHalf", f"{g9[(0.5, 'threshold')][0]:.1f}")

# --- E10 pricing ---
e10 = load("e10_pricing")
g10 = rel_by(e10, lambda m: (m["_kw"]["pricing"], m["_kw"]["seed"]), pol, cost)
rows = []
for pr in ["fixed", "adaptive"]:
    for p, name in [("cheapest", "Cheapest"), ("threshold", "Threshold"), ("rem", "\\textbf{REM}")]:
        sel = [m for m in e10 if m["_kw"]["pricing"] == pr and m["_kw"]["policy"] == p]
        c = ci([m["cost_per_useful_gpu_h"] for m in sel])
        dl = "---" if p == "cheapest" else f"{g10[(pr, p)][0]:+.1f}".replace("-", "$-$")
        if pr == "adaptive":
            rp = {t: np.mean([m["pricing"]["final_rel_price"][t] for m in sel]) for t in ["community", "premium"]}
            rv = np.mean([m["pricing"]["revenue_per_host"]["premium"] for m in sel])
            cr = np.mean([m["pricing"]["corr_loglam_logprice_comm"] for m in sel])
            extra = f"{rp['community']:.2f} & {rp['premium']:.2f} & {cr:.2f}".replace("-", "$-$")
        else:
            extra = "1.00 & 1.00 & ---"
        rows.append(f"{pr.capitalize() if p == 'cheapest' else ''} & {name} & {c[0]:.3f} & {dl} & {extra} \\\\")
    if pr == "fixed":
        rows.append("\\midrule")
write_tab("tab_pricing.tex", "llrrrrr",
          "Prices & Policy & Cost (\\$) & $\\Delta$ (\\%) & Comm. & Prem. & $r(\\lambda,p)$", rows)
rmac("PrRemAd", f"{abs(g10[('adaptive', 'rem')][0]):.1f}")
rmac("PrRemFix", f"{abs(g10[('fixed', 'rem')][0]):.1f}")
rmac("PrThrAd", f"{g10[('adaptive', 'threshold')][0]:.0f}")
rmac("PrThrFix", f"{g10[('fixed', 'threshold')][0]:.0f}")


def e10m(pr, p, f):
    return np.mean([f(m) for m in e10 if m["_kw"]["pricing"] == pr and m["_kw"]["policy"] == p])


rmac("PrCorrRem", f"{e10m('adaptive', 'rem', lambda m: m['pricing']['corr_loglam_logprice_comm']):.2f}".replace("-", "$-$"))
rmac("PrCorrCheap", f"{e10m('adaptive', 'cheapest', lambda m: m['pricing']['corr_loglam_logprice_comm']):.2f}".replace("-", "$-$"))
rmac("PrRevPremRem", f"{e10m('adaptive', 'rem', lambda m: m['pricing']['revenue_per_host']['premium']):.0f}")
rmac("PrRevPremCheap", f"{e10m('adaptive', 'cheapest', lambda m: m['pricing']['revenue_per_host']['premium']):.0f}")
rmac("PrRevCommThr", f"{e10m('adaptive', 'threshold', lambda m: m['pricing']['revenue_per_host']['community']):.0f}")
rmac("PrRevCommCheap", f"{e10m('adaptive', 'cheapest', lambda m: m['pricing']['revenue_per_host']['community']):.0f}")
rmac("PrCommRelCheap", f"{e10m('adaptive', 'cheapest', lambda m: m['pricing']['final_rel_price']['community']):.2f}")
rmac("PrCommRelRem", f"{e10m('adaptive', 'rem', lambda m: m['pricing']['final_rel_price']['community']):.2f}")
rmac("PrHorizon", f"{np.mean([m['horizon_h'] for m in e10 if m['_kw']['pricing'] == 'adaptive']):.0f}")

# --- E4 reframing: capacity, not only spend ---
rmac("MigUtilGap", f"{100 * (np.mean([m['utilization'] for m in e4 if not m['_kw']['mig'] and m['_kw']['load'] == 0.5]) - np.mean([m['utilization'] for m in e4 if m['_kw']['mig'] and m['_kw']['load'] == 0.5])):.0f}")

with open(os.path.join(OUT, "numbers.tex"), "a") as f:
    for k, v in sorted(rev.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(json.dumps(rev, indent=0))

# --- V2 gang validation ---
vg = load("validation_gang")
with open(os.path.join(OUT, "numbers.tex"), "a") as f:
    f.write(f"\\newcommand{{\\GangValInCI}}{{{sum(abs(r['sim_mean'] - r['exact']) <= r['sim_ci'] for r in vg)}}}\n")
    f.write(f"\\newcommand{{\\GangValPoints}}{{{len(vg)}}}\n")
    f.write(f"\\newcommand{{\\GangValMaxErr}}{{{100 * max(abs(r['sim_mean'] - r['exact']) / r['exact'] for r in vg):.2f}}}\n")
    f.write(f"\\newcommand{{\\SimLines}}{{{sum(1 for _ in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim.py')))}}}\n")
