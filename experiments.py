"""Runs every experiment in the paper and writes results/*.json.

Usage: python experiments.py [name ...]   (default: all)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from sim import (Config, JobClass, Market, TierSpec, exact_expected_paid,
                 expected_time, simulate)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(8))
N_JOBS = 5000


def _run(kw):
    t0 = time.time()
    m = simulate(**kw)
    m["_kw"] = {k: ([t.__dict__ for t in v] if k in ("tiers", "jobs") else v) for k, v in kw.items()}
    m["_secs"] = time.time() - t0
    return m


def run_grid(name, configs, procs=2):
    t0 = time.time()
    with Pool(procs) as p:
        res = p.map(_run, configs, chunksize=1)
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(res, f)
    print(f"[{name}] {len(configs)} runs in {time.time() - t0:.0f}s", flush=True)
    return res


# ---------------------------------------------------------------------------
# V1: simulator vs closed-form expected occupied time (single host, one job)
# ---------------------------------------------------------------------------
def _single(args):
    mtbf, tau, seed = args
    cfg = Config(
        n_hosts=1, n_jobs=1, load=0.01, policy="cheapest", ckpt="fixed",
        fixed_tau=tau, mig=False, reputation="oracle", warmup_frac=0.0,
        tiers=[TierSpec("one", 1.0, 1.0, 1.0, mtbf, 0.0, 0.5)],
        jobs=[JobClass("ref", 1.0, 7, 12.0, 0.0, 0.10, 0.50)], seed=seed)
    mk = Market(cfg)
    mk.run()
    return mk.jobs[0].paid_time


def exp_validation():
    W, C, R = 12.0, 0.10, 0.50
    rows = []
    reps = 3000
    for mtbf in [2, 3, 4.5, 8, 16, 32]:
        for tau in [0.5, 1.0, 2.0]:
            with Pool(2) as p:
                xs = np.array(p.map(_single, [(mtbf, tau, s) for s in range(reps)], chunksize=100))
            lam = 1.0 / mtbf
            ex = exact_expected_paid(W, tau, C, R, lam)
            dl = expected_time(W, tau, C, R, lam)
            n_ck = math.ceil(W / tau) - 1
            rows.append(dict(
                mtbf=mtbf, tau=tau, sim_mean=float(xs.mean()),
                sim_ci=float(1.96 * xs.std(ddof=1) / math.sqrt(reps)),
                exact=ex, daly=dl,
                sim_overhead=float((xs.mean() - R - W - n_ck * C) / W),
                exact_overhead=float((ex - R - W - n_ck * C) / W)))
            print(rows[-1], flush=True)
    with open(os.path.join(OUT, "validation.json"), "w") as f:
        json.dump(rows, f)


# ---------------------------------------------------------------------------
# E1: matching policies across offered load
# ---------------------------------------------------------------------------
E1_POLICIES = ["random", "cheapest", "verified_cheapest", "threshold",
               "most_reliable", "rem", "rem_oracle"]
E1_LOADS = [0.2, 0.35, 0.5, 0.65, 0.8]


def exp_policies():
    cfgs = []
    for load in E1_LOADS:
        for pol in E1_POLICIES:
            for s in SEEDS:
                # filtering to non-community hosts halves usable capacity:
                # beyond ~0.45 offered load that queue is unstable, so fewer
                # jobs/seeds keep run time bounded (marked in the paper)
                if pol == "verified_cheapest" and load > 0.45:
                    if s >= 3:
                        continue
                    cfgs.append(dict(policy=pol, load=load, seed=s, n_jobs=2500))
                    continue
                cfgs.append(dict(policy=pol, load=load, seed=s, n_jobs=N_JOBS))
    run_grid("e1_policies", cfgs)


# ---------------------------------------------------------------------------
# E2: when does reliability-aware matching pay?  community MTBF x price
# ---------------------------------------------------------------------------
E2_MTBF = [1.5, 3.0, 4.5, 8.0, 16.0]
E2_PRICE = [0.6, 0.9, 1.1, 1.3, 1.5]   # midpoint $/GPU-h of community tier


def _tiers(mtbf_c, price_mid):
    return [TierSpec("community", 0.50, price_mid - 0.2, price_mid + 0.2, mtbf_c, 0.8, 2.0),
            TierSpec("verified", 0.35, 1.50, 1.87, 16.0, 0.6, 1.0),
            TierSpec("premium", 0.15, 2.10, 2.99, 200.0, 0.5, 0.5)]


def exp_sensitivity():
    cfgs = []
    for mt in E2_MTBF:
        for pm in E2_PRICE:
            for pol in ["cheapest", "threshold", "rem"]:
                for s in SEEDS[:5]:
                    cfgs.append(dict(policy=pol, load=0.35, seed=s, n_jobs=N_JOBS,
                                     tiers=_tiers(mt, pm)))
    run_grid("e2_sensitivity", cfgs)


# ---------------------------------------------------------------------------
# E3: checkpoint-interval ablation
# ---------------------------------------------------------------------------
def exp_checkpoint():
    cfgs = []
    for ck, tau in [("fixed", 0.5), ("fixed", 1.0), ("fixed", 2.0), ("daly", None)]:
        for pol in ["cheapest", "rem"]:
            for s in SEEDS:
                kw = dict(policy=pol, load=0.35, seed=s, n_jobs=N_JOBS, ckpt=ck)
                if tau:
                    kw["fixed_tau"] = tau
                cfgs.append(kw)
    run_grid("e3_checkpoint", cfgs)


# ---------------------------------------------------------------------------
# E4: MIG-style slicing vs whole-GPU allocation
# ---------------------------------------------------------------------------
def exp_mig():
    cfgs = []
    for load in [0.2, 0.35, 0.5, 0.65, 0.8]:
        for mig in [True, False]:
            for s in SEEDS:
                cfgs.append(dict(policy="rem", load=load, seed=s, n_jobs=N_JOBS, mig=mig))
    run_grid("e4_mig", cfgs)


# ---------------------------------------------------------------------------
# E5: reputation signal quality
# ---------------------------------------------------------------------------
def exp_reputation():
    cfgs = []
    # (placement signal, checkpoint signal, observed history before t=0)
    variants = [("tier", None, 0.0), ("tier", "learned", 72.0),
                ("learned", None, 0.0), ("learned", None, 24.0),
                ("learned", None, 72.0), ("learned", None, 336.0), ("oracle", None, 0.0)]
    for rep, ck, hist in variants:
        for s in SEEDS:
            kw = dict(policy="rem", load=0.35, seed=s, n_jobs=N_JOBS,
                      reputation=rep, history_hours=hist)
            if ck:
                kw["ckpt_reputation"] = ck
            cfgs.append(kw)
    run_grid("e5_reputation", cfgs)


EXPS = dict(validation=exp_validation, policies=exp_policies,
            sensitivity=exp_sensitivity, checkpoint=exp_checkpoint,
            mig=exp_mig, reputation=exp_reputation)



# ===========================================================================
# Revision experiments (E6-E10)
# ===========================================================================
from sim import DEFAULT_JOBS  # noqa: E402

SEEDS6 = list(range(6))


def _tiers_sigma(s):
    """Same mean failure rate per tier, host-to-host spread sigma = s."""
    out = []
    for t in [TierSpec(**x.__dict__) for x in __import__("sim").DEFAULT_TIERS]:
        if t.name in ("community", "verified"):
            t.mtbf_median = t.mtbf_median * math.exp((s ** 2 - t.mtbf_sigma ** 2) / 2)
            t.mtbf_sigma = s
        out.append(t)
    return out


E6_SIGMA = [0.0, 0.4, 0.8, 1.2]
# placement and checkpoint signals are varied separately so that the value of
# reputation for host choice is not confused with its value for checkpointing
E6_POL = [("cheapest", {}), ("threshold", {}),
          ("cheapest", {"reputation": "tier"}),                       # tier-level checkpoints
          ("rem", {"reputation": "tier", "ckpt_reputation": "learned"}),  # tier-level placement
          ("rem", {"reputation": "tier"}),                            # both tier-level
          ("rem", {}), ("rem_oracle", {})]


def exp_sigma():
    cfgs = []
    for s in E6_SIGMA:
        for pol, extra in E6_POL:
            for sd in SEEDS6:
                cfgs.append(dict(policy=pol, load=0.35, seed=sd, n_jobs=N_JOBS,
                                 tiers=_tiers_sigma(s), **extra))
    run_grid("e6_sigma", cfgs)


def exp_failmodel():
    cfgs = []
    for k in [0.7, 1.0, 1.5]:
        for c in [0.0, 0.5]:
            for pol in ["cheapest", "threshold", "rem"]:
                for sd in SEEDS6:
                    cfgs.append(dict(policy=pol, load=0.35, seed=sd, n_jobs=N_JOBS,
                                     weibull_k=k, corr_frac=c))
    run_grid("e7_failmodel", cfgs)


def gang_jobs():
    js = [JobClass(**j.__dict__) for j in DEFAULT_JOBS]
    js[0].prob = 0.12                              # single-GPU training
    js.append(JobClass("multinode", 0.03, 7, 8.0, 0.8, 0.20, 1.00, hosts=4))
    return js


def exp_gang():
    cfgs = []
    for load in [0.2, 0.35, 0.5]:
        for c in [0.0, 0.5]:
            for pol in ["cheapest", "threshold", "most_reliable", "rem"]:
                for sd in SEEDS6:
                    cfgs.append(dict(policy=pol, load=load, seed=sd, n_jobs=N_JOBS,
                                     corr_frac=c, jobs=gang_jobs()))
    run_grid("e8_gang", cfgs)


def exp_scale():
    cfgs = []
    # loads at which every policy's queue is stable at this scale; at higher
    # load the threshold policy's queue grows without bound (cf. E1)
    for load in [0.35, 0.5]:
        for pol in ["cheapest", "threshold", "rem"]:
            for sd in range(3):
                cfgs.append(dict(policy=pol, load=load, seed=sd, n_hosts=1000, n_jobs=25000))
    run_grid("e9_scale", cfgs)


def exp_pricing():
    cfgs = []
    for pricing in ["fixed", "adaptive"]:
        for pol in ["cheapest", "threshold", "rem"]:
            for sd in SEEDS6:
                cfgs.append(dict(policy=pol, load=0.35, seed=sd, n_jobs=10000, pricing=pricing))
    run_grid("e10_pricing", cfgs)


EXPS.update(sigma=exp_sigma, failmodel=exp_failmodel, gang=exp_gang,
            scale=exp_scale, pricing=exp_pricing)


# ---------------------------------------------------------------------------
# V2: gang job (4 hosts) vs closed form with superposed failure rate
# ---------------------------------------------------------------------------
def _single_gang(args):
    mtbf, seed = args
    cfg = Config(
        n_hosts=4, n_jobs=1, load=0.01, policy="cheapest", ckpt="fixed",
        fixed_tau=1.0, mig=False, reputation="oracle", warmup_frac=0.0,
        tiers=[TierSpec("one", 1.0, 1.0, 1.0, mtbf, 0.0, 0.5)],
        jobs=[JobClass("g", 1.0, 7, 12.0, 0.0, 0.10, 0.50, hosts=4)], seed=seed)
    mk = Market(cfg)
    mk.run()
    return mk.jobs[0].paid_time


def exp_validation_gang():
    W, C, R, reps = 12.0, 0.10, 0.50, 3000
    rows = []
    for mtbf in [16.0, 32.0, 64.0]:
        with Pool(2) as p:
            xs = np.array(p.map(_single_gang, [(mtbf, s) for s in range(reps)], chunksize=100))
        ex = exact_expected_paid(W, 1.0, C, R, 4.0 / mtbf)
        rows.append(dict(mtbf=mtbf, sim_mean=float(xs.mean()),
                         sim_ci=float(1.96 * xs.std(ddof=1) / math.sqrt(reps)), exact=ex))
        print(rows[-1], flush=True)
    with open(os.path.join(OUT, "validation_gang.json"), "w") as f:
        json.dump(rows, f)


EXPS.update(validation_gang=exp_validation_gang)

if __name__ == "__main__":
    names = sys.argv[1:] or list(EXPS)
    for n in names:
        EXPS[n]()
