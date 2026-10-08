"""Builds the paper's figures (PDF) from results/*.json."""
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from analyze import ci, load

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd7"
# fixed categorical order (reference palette, light mode); color follows the policy
SERIES = {
    "rem": ("REM (ours)", "#2a78d6", "o", "-"),
    "cheapest": ("Cheapest", "#eb6834", "s", "--"),
    "threshold": ("Threshold (4 h)", "#1baf7a", "^", "-."),
    "verified_cheapest": ("Verified-only", "#eda100", "D", ":"),
    "most_reliable": ("Most-reliable", "#e87ba4", "v", (0, (5, 1, 1, 1))),
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def paired(res, keyf, pol, base, metric):
    d = defaultdict(dict)
    for m in res:
        d[keyf(m)][m["_kw"]["policy"]] = m
    out = defaultdict(list)
    for k, v in d.items():
        if pol in v and base in v:
            out[k[0]].append(100 * (metric(v[pol]) / metric(v[base]) - 1))
    return {k: ci(v) for k, v in out.items()}


def fig_policies():
    e1 = load("e1_policies")
    keyf = lambda m: (m["_kw"]["load"], m["_kw"]["seed"])
    cost = lambda m: m["cost_per_useful_gpu_h"]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.35))
    ax = axs[0]
    for pol in ["most_reliable", "verified_cheapest", "threshold", "rem"]:
        lab, c, mk, ls = SERIES[pol]
        g = paired(e1, keyf, pol, "cheapest", cost)
        xs = sorted(g)
        ax.errorbar(xs, [g[x][0] for x in xs], yerr=[g[x][1] for x in xs], color=c,
                    marker=mk, ms=4, lw=1.4, ls=ls, capsize=2, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.text(0.425, 1.2, "Cheapest-first baseline", color=INK2, fontsize=6.5, va="bottom", ha="center")
    ax.set_xlabel("Offered load")
    ax.set_ylabel("Cost per useful GPU-hour\nvs. Cheapest (%)")
    ax.set_title("(a) Buyer cost relative to cheapest-first")
    style(ax)
    ax.legend(frameon=False, loc="upper right", ncol=1)

    ax = axs[1]
    for pol in ["cheapest", "rem", "threshold", "verified_cheapest"]:
        lab, c, mk, ls = SERIES[pol]
        g = defaultdict(list)
        for m in e1:
            if m["_kw"]["policy"] == pol:
                g[m["_kw"]["load"]].append(m["wait_mean"] * 60)
        xs = sorted(g)
        mu = [max(ci(g[x])[0], 0.05) for x in xs]
        ax.plot(xs, mu, color=c, marker=mk, ms=4, lw=1.4, ls=ls, label=lab)
    ax.set_yscale("log")
    ax.set_ylim(0.04, 3000)
    ax.set_xlabel("Offered load")
    ax.set_ylabel("Mean queueing delay (min, log)")
    ax.set_title("(b) Waiting for an eligible host")
    style(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(FIG, "policies.pdf"))
    fig.savefig(os.path.join(FIG, "policies.png"), dpi=200)


def fig_sensitivity():
    e2 = load("e2_sensitivity")
    d = defaultdict(dict)
    for m in e2:
        t = m["_kw"]["tiers"][0]
        key = (t["mtbf_median"], round((t["price_lo"] + t["price_hi"]) / 2, 2), m["_kw"]["seed"])
        d[key][m["_kw"]["policy"]] = m["cost_per_useful_gpu_h"]
    mtbfs = sorted({k[0] for k in d})
    prices = sorted({k[1] for k in d})
    seeds = sorted({k[2] for k in d})

    def grid(a, b):
        return np.array([[np.mean([100 * (1 - d[(mt, pm, s)][a] / d[(mt, pm, s)][b]) for s in seeds])
                          for pm in prices] for mt in mtbfs])

    blues = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#9cc3ef", "#2a78d6", "#123f78"])
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.45))
    for ax, (a, b, title) in zip(axs, [("rem", "cheapest", "(a) REM saving vs. Cheapest (%)"),
                                        ("rem", "threshold", "(b) REM saving vs. Threshold (%)")]):
        G = grid(a, b)
        vmax = max(1e-9, G.max())
        im = ax.imshow(G, cmap=blues, vmin=0, vmax=vmax, aspect="auto", origin="lower")
        for i in range(len(mtbfs)):
            for j in range(len(prices)):
                v = G[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.55 * vmax else INK)
        ax.set_xticks(range(len(prices)), [f"${p:.2f}" for p in prices])
        ax.set_yticks(range(len(mtbfs)), [f"{m:g}" for m in mtbfs])
        ax.set_xlabel("Community-tier mean price ($/GPU-h)")
        ax.set_ylabel("Community-tier median MTBF (h)")
        ax.set_title(title)
        ax.add_patch(plt.Rectangle((prices.index(1.1) - 0.5, mtbfs.index(4.5) - 0.5), 1, 1,
                                   fill=False, ec=INK, lw=1.2))
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(FIG, "sensitivity.pdf"))
    fig.savefig(os.path.join(FIG, "sensitivity.png"), dpi=200)
    return mtbfs, prices


def fig_mig():
    e4 = load("e4_mig")
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.25))
    lab = {True: ("MIG-style slices", "#2a78d6", "o", "-"),
           False: ("Whole-GPU allocation", "#eb6834", "s", "--")}
    for ax, metric, ylab, title, logy in [
        (axs[0], lambda m: m["total_spend"] / 1000, "Total buyer spend ($k)", "(a) Spend for the same workload", False),
        (axs[1], lambda m: m["by_class"]["small"]["wait"] * 60, "Small-job mean wait (min, log)", "(b) Queueing delay of 1/7-GPU jobs", True)]:
        for mig in (True, False):
            name, c, mk, ls = lab[mig]
            g = defaultdict(list)
            for m in e4:
                if m["_kw"]["mig"] == mig:
                    g[m["_kw"]["load"]].append(metric(m))
            xs = sorted(g)
            mu = [ci(g[x])[0] for x in xs]
            er = [ci(g[x])[1] for x in xs]
            if logy:
                mu = [max(v, 0.05) for v in mu]
                ax.plot(xs, mu, color=c, marker=mk, ms=4, lw=1.4, ls=ls, label=name)
            else:
                ax.errorbar(xs, mu, yerr=er, color=c, marker=mk, ms=4, lw=1.4, ls=ls, capsize=2, label=name)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(0.04, 3000)
        ax.set_xlabel("Offered load")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        style(ax)
        ax.legend(frameon=False, loc="lower right" if not logy else "upper left")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(FIG, "mig.pdf"))
    fig.savefig(os.path.join(FIG, "mig.png"), dpi=200)


if __name__ == "__main__":
    fig_policies()
    fig_sensitivity()
    fig_mig()
    print("ok")


# ===========================================================================
# Revision figures
# ===========================================================================
VIOLET = "#4a3aa7"


def fig_sigma():
    e6 = load("e6_sigma")
    d = defaultdict(dict)
    for m in e6:
        kw = m["_kw"]
        s = kw["tiers"][0]["mtbf_sigma"]
        rep = kw.get("reputation", "learned")
        ck = kw.get("ckpt_reputation") or rep
        d[(s, kw["seed"])][(kw["policy"], rep, ck)] = m["cost_per_useful_gpu_h"]
    sig = sorted({k[0] for k in d})
    seeds = sorted({k[1] for k in d})
    base = ("cheapest", "learned", "learned")
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    series = [(("rem", "learned", "learned"), "REM, per-host reputation", "#2a78d6", "o", "-"),
              (("rem", "tier", "learned"), "REM, tier average for placement", VIOLET, "s", "--"),
              (("cheapest", "tier", "tier"), "Cheapest, tier average for checkpoints", "#eb6834", "^", "-."),
              (("rem_oracle", "learned", "learned"), "REM, true failure rates", INK2, None, ":")]
    for key, lab, c, mk, ls in series:
        mu, er = [], []
        for s in sig:
            m_, h_ = ci([100 * (d[(s, sd)][key] / d[(s, sd)][base] - 1) for sd in seeds])
            mu.append(m_)
            er.append(h_)
        ax.errorbar(sig, mu, yerr=er, color=c, marker=mk, ms=4, lw=1.4, ls=ls, capsize=2, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(sig)
    ax.set_xlabel(r"Host-to-host reliability spread $\sigma$ (same tier mean)")
    ax.set_ylabel("Cost vs. Cheapest (%)")
    style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=6.2)
    ax.set_ylim(-2.7, 4.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sigma.pdf"))
    fig.savefig(os.path.join(FIG, "sigma.png"), dpi=200)


def fig_gang():
    e8 = load("e8_gang")
    d = defaultdict(dict)
    for m in e8:
        kw = m["_kw"]
        if kw["corr_frac"] != 0.0:
            continue
        d[(kw["load"], kw["seed"])][kw["policy"]] = m
    loads = sorted({k[0] for k in d})
    seeds = sorted({k[1] for k in d})
    pols = ["cheapest", "threshold", "most_reliable", "rem"]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.3))
    width = 0.19
    for ax, (metric, ylab, title) in zip(axs, [
            (lambda m: m["by_class"]["multinode"]["cost"], "Cost per useful GPU-hour ($)", "(a) Cost of 4-GPU gang jobs"),
            (lambda m: m["by_class"]["multinode"]["jct"], "Mean completion time (h)", "(b) Completion time of 4-GPU gang jobs")]):
        for i, p in enumerate(pols):
            lab, c, _, _ = SERIES[p]
            xs = np.arange(len(loads)) + (i - 1.5) * (width + 0.02)
            vals = [ci([metric(d[(L, s)][p]) for s in seeds]) for L in loads]
            ax.bar(xs, [v[0] for v in vals], width, color=c, label=lab, zorder=2)
            ax.errorbar(xs, [v[0] for v in vals], yerr=[v[1] for v in vals], fmt="none",
                        ecolor=INK, elinewidth=0.7, capsize=1.5, zorder=3)
        ax.set_xticks(range(len(loads)), [rf"$\rho={L:g}$" for L in loads])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        style(ax)
        ax.grid(axis="x", visible=False)
    axs[0].legend(frameon=False, loc="upper right", ncol=2, fontsize=6.5)
    axs[0].set_ylim(0, axs[0].get_ylim()[1] * 1.18)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(FIG, "gang.pdf"))
    fig.savefig(os.path.join(FIG, "gang.png"), dpi=200)


if __name__ == "__main__":
    fig_sigma()
    fig_gang()
    print("revision figures ok")
