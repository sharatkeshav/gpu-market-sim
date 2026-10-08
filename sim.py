"""
Discrete-event simulator of a GPU-sharing marketplace.

Each listing ("host") offers one GPU that can be partitioned into 7 MIG-style
slice units. Hosts fail and recover; up times are exponential by default, or
Weibull, and a fraction of failures can be correlated across the hosts of one
seller. Jobs arrive as a Poisson process, request s in {1,2,3,7} slice units
(or several whole GPUs on distinct hosts, gang-scheduled) and a fixed amount
of useful work W, checkpoint every tau hours of work at cost C, and pay a
setup/restart cost R at every (re)start.

A matching policy chooses which free host(s) a queued job runs on. Buyers pay
the posted price, pro-rated by slice size, for all occupied time (useful work,
checkpoint writes, restarts and work lost to failures). Prices are fixed by
default, or adjusted periodically by sellers.

With all extensions at their defaults the simulator reproduces the original
model exactly (same random draws in the same order).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

UNITS = 7  # MIG-style slice units per GPU


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class TierSpec:
    name: str
    share: float          # fraction of listings
    price_lo: float       # $/GPU-hour, uniform range
    price_hi: float
    mtbf_median: float    # hours; host MTBF ~ lognormal(median, sigma)
    mtbf_sigma: float
    repair_mean: float    # hours down after a failure


@dataclass
class JobClass:
    name: str
    prob: float
    units: int            # slice units requested per GPU
    work_median: float    # hours of useful work at this size
    work_sigma: float
    ckpt_cost: float      # hours per checkpoint write (C)
    restart: float        # hours of setup on every (re)start (R)
    hosts: int = 1        # >1: gang-scheduled on this many whole GPUs


DEFAULT_TIERS = [
    TierSpec("community", 0.50, 0.90, 1.30, 4.5, 0.8, 2.0),
    TierSpec("verified", 0.35, 1.50, 1.87, 16.0, 0.6, 1.0),
    TierSpec("premium", 0.15, 2.10, 2.99, 200.0, 0.5, 0.5),
]

DEFAULT_JOBS = [
    JobClass("train", 0.15, 7, 10.0, 0.8, 0.10, 0.50),
    JobClass("finetune", 0.25, 7, 2.5, 0.8, 0.05, 0.25),
    JobClass("batch_infer", 0.30, 3, 1.0, 0.8, 0.02, 0.10),
    JobClass("small", 0.30, 1, 0.4, 0.8, 0.01, 0.05),
]


@dataclass
class Config:
    n_hosts: int = 200
    n_jobs: int = 6000
    load: float = 0.6                 # offered load (useful GPU-work / capacity)
    policy: str = "rem"               # see POLICIES
    ckpt: str = "daly"                # "fixed" or "daly"
    fixed_tau: float = 1.0            # hours, used when ckpt == "fixed"
    threshold_hours: float = 4.0      # "threshold" policy: long jobs -> verified
    mig: bool = True                  # False: every job occupies a whole GPU
    reputation: str = "learned"       # placement signal: "learned", "tier", "oracle"
    ckpt_reputation: str | None = None  # checkpoint signal; None = same as placement
    history_hours: float = 72.0       # observed host history before t=0
    prior_alpha: float = 2.0          # Gamma prior on failure rate
    prior_mean_rate: float = 0.1      # failures / hour (platform-wide prior)
    warmup_frac: float = 0.1
    # --- failure-model extensions -------------------------------------
    weibull_k: float = 1.0            # up-time shape; 1.0 = exponential
    corr_frac: float = 0.0            # share of community failure rate that is seller-wide
    group_size: int = 5               # community hosts per seller
    # --- seller pricing -----------------------------------------------
    pricing: str = "fixed"            # "fixed" or "adaptive"
    price_period: float = 6.0         # hours between price updates
    price_step: float = 0.05          # multiplicative step
    price_band: tuple = (0.5, 1.5)    # bounds relative to the initial price
    tiers: list = field(default_factory=lambda: [TierSpec(**t.__dict__) for t in DEFAULT_TIERS])
    jobs: list = field(default_factory=lambda: [JobClass(**j.__dict__) for j in DEFAULT_JOBS])
    seed: int = 0


POLICIES = ["random", "cheapest", "verified_cheapest", "threshold", "most_reliable", "rem", "rem_oracle"]

# reliability weights tried by REM when choosing a set of hosts for a gang job
GANG_WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, math.inf]


# --------------------------------------------------------------------------
# Analytic model (Daly 2006, exponential failures, periodic checkpoints)
# --------------------------------------------------------------------------
def expected_time(work: float, tau: float, ckpt: float, restart: float, lam: float) -> float:
    """Expected wall time to finish `work` hours, including an initial setup R.

    Uses the first-order Daly model: each segment of tau hours of work plus a
    checkpoint of C hours is retried until it completes without a failure,
    and every failure costs a restart R.
    """
    if lam <= 1e-12:
        n_ck = max(math.ceil(work / tau) - 1, 0)
        return restart + work + n_ck * ckpt
    seg = (1.0 / lam) * math.exp(lam * restart) * math.expm1(lam * (tau + ckpt))
    return restart + seg * work / tau


def exact_expected_paid(work: float, tau: float, ckpt: float, restart: float, lam: float) -> float:
    """Exact expected occupied time under the simulator's protocol.

    Protocol: setup R at every (re)start; work proceeds in segments of tau
    hours, each followed by a checkpoint of C hours except the last; a failure
    anywhere loses the in-progress segment and triggers a new setup R.
    With exponential failures the setup costs (e^{lam R}-1)/lam and a segment
    of length L costs e^{lam R}(e^{lam L}-1)/lam in expectation.
    """
    n = max(math.ceil(work / tau - 1e-9), 1)
    lens = [tau + ckpt] * (n - 1) + [work - (n - 1) * tau]
    setup = math.expm1(lam * restart) / lam
    return setup + sum(math.exp(lam * restart) * math.expm1(lam * L) / lam for L in lens)


def daly_tau(ckpt: float, lam: float, lo: float = 0.05, hi: float = 24.0) -> float:
    """Young/Daly first-order optimal checkpoint interval sqrt(2C/lambda)."""
    if lam <= 1e-12:
        return hi
    return float(min(max(math.sqrt(2.0 * ckpt / lam), lo), hi))


# --------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------
class Host:
    __slots__ = ("hid", "tier", "price", "base_price", "lam", "lam_ind", "repair",
                 "up", "free", "jobs", "obs_fail", "obs_up", "up_since",
                 "busy_unit_time", "last_change", "fail_ver", "group",
                 "busy_at_update", "revenue")

    def __init__(self, hid, tier, price, lam, repair):
        self.hid, self.tier, self.price, self.lam, self.repair = hid, tier, price, lam, repair
        self.base_price = price
        self.lam_ind = lam        # rate of independent failures
        self.up = True
        self.free = UNITS
        self.jobs = set()
        self.obs_fail = 0.0
        self.obs_up = 0.0
        self.up_since = 0.0
        self.busy_unit_time = 0.0
        self.last_change = 0.0
        self.fail_ver = 0
        self.group = None
        self.busy_at_update = 0.0
        self.revenue = 0.0


class Job:
    __slots__ = ("jid", "cls", "units", "req_units", "gpus", "work", "remaining",
                 "C", "R", "arrive", "start", "finish", "hosts", "tau", "version",
                 "paid_time", "paid_dollars", "ckpt_time", "n_fail",
                 "attempt_start", "first_start", "tier_work", "rates")

    def __init__(self, jid, cls, units, work, C, R, arrive, gpus=1):
        self.jid, self.cls, self.units, self.work = jid, cls, units, work
        self.req_units = units
        self.gpus = gpus
        self.remaining = work
        self.C, self.R, self.arrive = C, R, arrive
        self.start = None
        self.finish = None
        self.hosts = []
        self.tau = None
        self.version = 0
        self.paid_time = 0.0
        self.paid_dollars = 0.0
        self.ckpt_time = 0.0
        self.n_fail = 0
        self.attempt_start = None
        self.first_start = None
        self.tier_work = {}
        self.rates = []


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------
class Market:
    ARRIVAL, JOB_END, HOST_FAIL, HOST_REPAIR, GROUP_FAIL, PRICE_UPDATE = 0, 1, 2, 3, 4, 5

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0.0
        self.events = []
        self._seq = 0
        self.queue = []          # job ids in arrival order
        self.jobs = []
        self.groups = []
        self._build_hosts()
        self._build_workload()
        if cfg.pricing == "adaptive":
            self._push(cfg.price_period, self.PRICE_UPDATE, 0)

    # ---- setup -----------------------------------------------------------
    def _build_hosts(self):
        cfg, rng = self.cfg, self.rng
        counts = np.round([t.share * cfg.n_hosts for t in cfg.tiers]).astype(int)
        counts[-1] = cfg.n_hosts - counts[:-1].sum()
        self.hosts = []
        self.tier_prior_rate = {}
        for t, n in zip(cfg.tiers, counts):
            mtbf = t.mtbf_median * np.exp(t.mtbf_sigma * rng.standard_normal(n))
            prices = rng.uniform(t.price_lo, t.price_hi, n)
            # tier-level prior: mean failure rate of the tier's distribution
            self.tier_prior_rate[t.name] = (1.0 / t.mtbf_median) * math.exp(t.mtbf_sigma ** 2 / 2)
            for i in range(n):
                h = Host(len(self.hosts), t.name, float(prices[i]), 1.0 / float(mtbf[i]), t.repair_mean)
                # pre-sample observed history for the reputation system
                h.obs_up = cfg.history_hours
                h.obs_fail = float(rng.poisson(h.lam * cfg.history_hours))
                self.hosts.append(h)
        # seller-correlated outages: consecutive community hosts form sellers
        self.group_rate = []
        if cfg.corr_frac > 0:
            comm = [h for h in self.hosts if h.tier == "community"]
            for g in range(0, len(comm), cfg.group_size):
                members = comm[g:g + cfg.group_size]
                gid = len(self.groups)
                rate = cfg.corr_frac * float(np.mean([h.lam for h in members]))
                for h in members:
                    h.group = gid
                    h.lam_ind = (1.0 - cfg.corr_frac) * h.lam
                    h.lam = h.lam_ind + rate     # true total failure rate
                self.groups.append(members)
                self.group_rate.append(rate)
        for h in self.hosts:
            self._push(self._life(h), self.HOST_FAIL, h.hid, h.fail_ver)
        for gid, rate in enumerate(self.group_rate):
            self._push(rng.exponential(1.0 / rate), self.GROUP_FAIL, gid)

    def _life(self, h: Host) -> float:
        """Sample an up time for host h (independent failures only)."""
        k = self.cfg.weibull_k
        if k == 1.0:
            return self.rng.exponential(1.0 / h.lam_ind)
        scale = (1.0 / h.lam_ind) / math.gamma(1.0 + 1.0 / k)
        return scale * self.rng.weibull(k)

    def _build_workload(self):
        cfg, rng = self.cfg, self.rng
        probs = np.array([j.prob for j in cfg.jobs])
        probs = probs / probs.sum()
        mean_gpu_work = sum(
            p * j.work_median * math.exp(j.work_sigma ** 2 / 2) * j.units / UNITS * j.hosts
            for p, j in zip(probs, cfg.jobs))
        self.arrival_rate = cfg.load * cfg.n_hosts / mean_gpu_work
        t = 0.0
        for i in range(cfg.n_jobs):
            t += rng.exponential(1.0 / self.arrival_rate)
            k = rng.choice(len(cfg.jobs), p=probs)
            jc = cfg.jobs[k]
            w = jc.work_median * math.exp(jc.work_sigma * rng.standard_normal())
            units = jc.units if (cfg.mig and jc.hosts == 1) else UNITS
            job = Job(i, jc.name, units, w, jc.ckpt_cost, jc.restart, t, gpus=jc.hosts)
            job.req_units = jc.units  # true slice need, used to normalise cost
            job.tier_work = {tt.name: 0.0 for tt in cfg.tiers}
            self.jobs.append(job)
            self._push(t, self.ARRIVAL, i)

    # ---- event plumbing --------------------------------------------------
    def _push(self, time, kind, ident, version=0):
        self._seq += 1
        heapq.heappush(self.events, (time, self._seq, kind, ident, version))

    # ---- reputation ------------------------------------------------------
    def rate_estimate(self, h: Host, mode: str | None = None) -> float:
        cfg = self.cfg
        mode = mode or cfg.reputation
        if mode == "oracle":
            return h.lam
        if mode == "tier":
            return self.tier_prior_rate[h.tier]
        up_now = h.obs_up + ((self.t - h.up_since) if h.up else 0.0)
        alpha0 = cfg.prior_alpha
        beta0 = alpha0 / cfg.prior_mean_rate
        return (alpha0 + h.obs_fail) / (beta0 + up_now)

    def _lam(self, h: Host) -> float:
        return h.lam if self.cfg.policy == "rem_oracle" else self.rate_estimate(h)

    def _lam_ckpt(self, h: Host) -> float:
        """Failure-rate estimate used to set the checkpoint interval."""
        if self.cfg.ckpt_reputation is None:
            return self._lam(h)
        return self.rate_estimate(h, self.cfg.ckpt_reputation)

    def tau_for(self, job: Job, lam_hat: float) -> float:
        if self.cfg.ckpt == "fixed":
            return self.cfg.fixed_tau
        return daly_tau(job.C, lam_hat)

    def _eligible(self, job: Job, pool):
        pol = self.cfg.policy
        need = job.units
        cands = [h for h in pool if h.up and h.free >= need]
        if pol == "verified_cheapest" or (
                pol == "threshold" and job.remaining > self.cfg.threshold_hours):
            cands = [h for h in cands if h.tier != "community"]
        return cands

    # ---- matching --------------------------------------------------------
    def choose_host(self, job: Job, avail=None):
        """Single-host placement (unchanged from the original model)."""
        pool = self.hosts if avail is None else avail
        cands = self._eligible(job, pool)
        pol = self.cfg.policy
        if not cands:
            return None
        if pol == "random":
            return cands[self.rng.integers(len(cands))]
        if pol in ("cheapest", "verified_cheapest", "threshold"):
            key = lambda h: (h.price, h.free)
        elif pol == "most_reliable":
            key = lambda h: (self.rate_estimate(h), h.price)
        elif pol in ("rem", "rem_oracle"):
            def key(h):
                lam = self._lam(h)
                tau = self.tau_for(job, lam)
                et = expected_time(job.remaining, tau, job.C, job.R, lam)
                return (h.price * job.units / UNITS * et, h.free)
        else:
            raise ValueError(pol)
        return min(cands, key=key)

    def gang_cost(self, job: Job, hs) -> float:
        lam = sum(self._lam(h) for h in hs)
        tau = self.tau_for(job, lam)
        return sum(h.price for h in hs) * expected_time(job.remaining, tau, job.C, job.R, lam)

    def choose_hosts(self, job: Job, avail):
        """Gang placement on job.gpus distinct, fully free hosts."""
        g = job.gpus
        cands = self._eligible(job, avail)
        if len(cands) < g:
            return None
        pol = self.cfg.policy
        if pol == "random":
            idx = self.rng.choice(len(cands), size=g, replace=False)
            return [cands[i] for i in idx]
        if pol in ("cheapest", "verified_cheapest", "threshold"):
            return sorted(cands, key=lambda h: (h.price, h.hid))[:g]
        if pol == "most_reliable":
            return sorted(cands, key=lambda h: (self.rate_estimate(h), h.price))[:g]
        if pol in ("rem", "rem_oracle"):
            lam = {h.hid: self._lam(h) for h in cands}
            best, best_c = None, math.inf
            for w in GANG_WEIGHTS:
                if math.isinf(w):
                    key = lambda h: (lam[h.hid], h.price)
                else:
                    key = lambda h, w=w: (h.price + w * lam[h.hid], h.hid)
                hs = sorted(cands, key=key)[:g]
                c = self.gang_cost(job, hs)
                if c < best_c:
                    best, best_c = hs, c
            return best
        raise ValueError(pol)

    def try_schedule(self):
        if not self.queue:
            return
        avail = [h for h in self.hosts if h.up and h.free > 0]
        if not avail:
            return
        max_free = max(h.free for h in avail)
        still = []
        for idx, jid in enumerate(self.queue):
            job = self.jobs[jid]
            if job.units > max_free:
                still.append(jid)
                continue
            if job.gpus == 1:
                h = self.choose_host(job, avail)
                hs = None if h is None else [h]
            else:
                hs = self.choose_hosts(job, avail)
            if hs is None:
                still.append(jid)
                continue
            self.start_job(job, hs)
            avail = [x for x in avail if x.free > 0]
            if not avail:
                still.extend(self.queue[idx + 1:])
                break
            max_free = max(x.free for x in avail)
        self.queue = still

    def _account_busy(self, h: Host):
        h.busy_unit_time += (UNITS - h.free) * (self.t - h.last_change)
        h.last_change = self.t

    def start_job(self, job: Job, hs):
        lam_hat = 0.0
        for h in hs:
            self._account_busy(h)
            h.free -= job.units
            h.jobs.add(job.jid)
            lam_hat += self._lam_ckpt(h)
        job.tau = self.tau_for(job, lam_hat)
        job.hosts = list(hs)
        job.rates = [h.price * job.units / UNITS for h in hs]
        job.attempt_start = self.t
        if job.first_start is None:
            job.first_start = self.t
        n_ck = max(math.ceil(job.remaining / job.tau - 1e-9) - 1, 0)
        dur = job.R + job.remaining + n_ck * job.C
        job.version += 1
        self._push(self.t + dur, self.JOB_END, job.jid, job.version)

    def _release(self, job: Job):
        for h in job.hosts:
            self._account_busy(h)
            h.free += job.units
            h.jobs.discard(job.jid)
        job.hosts = []

    def _charge(self, job: Job, elapsed: float, useful: float):
        job.paid_time += elapsed
        for h, r in zip(job.hosts, job.rates):
            job.paid_dollars += r * elapsed
            h.revenue += r * elapsed
            job.tier_work[h.tier] += useful / len(job.hosts)

    # ---- event handlers --------------------------------------------------
    def on_job_end(self, job: Job):
        elapsed = self.t - job.attempt_start
        n_ck = max(math.ceil(job.remaining / job.tau - 1e-9) - 1, 0)
        job.ckpt_time += n_ck * job.C
        self._charge(job, elapsed, job.remaining)
        job.remaining = 0.0
        job.finish = self.t
        self._release(job)

    def interrupt(self, job: Job):
        elapsed = self.t - job.attempt_start
        e = elapsed - job.R
        saved, k = 0.0, 0
        if e > 0:
            k = int(e // (job.tau + job.C))
            saved = min(k * job.tau, job.remaining)
        job.ckpt_time += k * job.C
        self._charge(job, elapsed, saved)
        job.remaining -= saved
        job.n_fail += 1
        job.version += 1  # invalidate pending JOB_END
        self._release(job)

    def on_host_fail(self, h: Host):
        # update observed reputation
        h.obs_up += self.t - h.up_since
        h.obs_fail += 1
        h.up = False
        h.fail_ver += 1
        victims = sorted(h.jobs)
        for jid in victims:
            self.interrupt(self.jobs[jid])
        # re-queue victims, preserving arrival order
        self.queue = sorted(set(self.queue) | set(victims), key=lambda j: self.jobs[j].arrive)
        self._push(self.t + self.rng.exponential(h.repair), self.HOST_REPAIR, h.hid)

    def on_host_repair(self, h: Host):
        self._account_busy(h)
        h.up = True
        h.up_since = self.t
        self._push(self.t + self._life(h), self.HOST_FAIL, h.hid, h.fail_ver)

    def on_group_fail(self, gid: int):
        for h in self.groups[gid]:
            if h.up:
                self.on_host_fail(h)
        self._push(self.t + self.rng.exponential(1.0 / self.group_rate[gid]), self.GROUP_FAIL, gid)

    def on_price_update(self):
        cfg = self.cfg
        for h in self.hosts:
            self._account_busy(h)
        util = {h.hid: (h.busy_unit_time - h.busy_at_update) / (UNITS * cfg.price_period)
                for h in self.hosts}
        # median reference: half the hosts raise and half cut, so prices do not
        # drift just because utilization is skewed across hosts
        u_bar = float(np.median(list(util.values())))
        lo, hi = cfg.price_band
        for h in self.hosts:
            u = util[h.hid]
            if u > u_bar + 0.02:
                h.price = min(h.price * (1 + cfg.price_step), hi * h.base_price)
            elif u < u_bar - 0.02:
                h.price = max(h.price * (1 - cfg.price_step), lo * h.base_price)
            h.busy_at_update = h.busy_unit_time
        self._push(self.t + cfg.price_period, self.PRICE_UPDATE, 0)

    # ---- main loop -------------------------------------------------------
    def run(self):
        done = 0
        n = self.cfg.n_jobs
        while self.events and done < n:
            time, _, kind, ident, version = heapq.heappop(self.events)
            self.t = time
            if kind == self.ARRIVAL:
                self.queue.append(ident)
            elif kind == self.JOB_END:
                job = self.jobs[ident]
                if version != job.version or job.finish is not None:
                    continue
                self.on_job_end(job)
                done += 1
            elif kind == self.HOST_FAIL:
                h = self.hosts[ident]
                if version != h.fail_ver or not h.up:
                    continue   # superseded by a seller-wide outage
                self.on_host_fail(h)
            elif kind == self.HOST_REPAIR:
                self.on_host_repair(self.hosts[ident])
            elif kind == self.GROUP_FAIL:
                self.on_group_fail(ident)
            elif kind == self.PRICE_UPDATE:
                self.on_price_update()
            self.try_schedule()
        for h in self.hosts:
            self._account_busy(h)
        return self.metrics()

    # ---- metrics ---------------------------------------------------------
    def metrics(self):
        cfg = self.cfg
        w0 = int(cfg.warmup_frac * cfg.n_jobs)
        js = [j for j in self.jobs[w0:] if j.finish is not None]
        # useful work is measured at the slice size the job actually needs, so
        # whole-GPU allocation of a small job counts its unused slices as waste
        useful_gpu = np.array([j.work * j.req_units / UNITS * j.gpus for j in js])
        dollars = np.array([j.paid_dollars for j in js])
        jct = np.array([j.finish - j.arrive for j in js])
        wait = np.array([j.first_start - j.arrive for j in js])
        interrupt_over = np.array([
            (j.paid_time - j.work - j.ckpt_time - j.R) / j.work for j in js])
        slowdown = jct / np.array([j.work for j in js])
        tier_work = {t.name: sum(j.tier_work[t.name] * j.req_units / UNITS * j.gpus for j in js)
                     for t in cfg.tiers}
        tot_tw = sum(tier_work.values())
        horizon = self.t
        util = sum(h.busy_unit_time for h in self.hosts) / (UNITS * len(self.hosts) * horizon)
        by_cls = {}
        for jc in cfg.jobs:
            sel = [i for i, j in enumerate(js) if j.cls == jc.name]
            if sel:
                by_cls[jc.name] = dict(
                    cost=float(dollars[sel].sum() / useful_gpu[sel].sum()),
                    jct=float(jct[sel].mean()),
                    wait=float(wait[sel].mean()),
                    overhead=float(interrupt_over[sel].mean()),
                    n=len(sel),
                )
        tier_over = {}
        for t in cfg.tiers:
            # interruption overhead of jobs that ran entirely on one tier
            sel = [i for i, j in enumerate(js)
                   if j.tier_work[t.name] > 0.999 * j.work]
            tier_over[t.name] = float(np.mean(interrupt_over[sel])) if sel else float("nan")
        out = dict(
            cost_per_useful_gpu_h=float(dollars.sum() / useful_gpu.sum()),
            total_spend=float(dollars.sum()),
            jct_mean=float(jct.mean()),
            jct_p95=float(np.percentile(jct, 95)),
            wait_mean=float(wait.mean()),
            slowdown_mean=float(slowdown.mean()),
            interrupt_overhead_mean=float(interrupt_over.mean()),
            failures_per_job=float(np.mean([j.n_fail for j in js])),
            utilization=float(util),
            tier_share={k: v / tot_tw for k, v in tier_work.items()},
            tier_overhead=tier_over,
            by_class=by_cls,
            n_done=len(js),
            horizon_h=float(horizon),
        )
        if cfg.pricing == "adaptive":
            comm = [h for h in self.hosts if h.tier == "community"]
            lam_c = np.log([h.lam for h in comm])
            rel_p = np.log([h.price / h.base_price for h in comm])
            out["pricing"] = dict(
                final_rel_price={t.name: float(np.mean([h.price / h.base_price for h in self.hosts
                                                        if h.tier == t.name])) for t in cfg.tiers},
                revenue_per_host={t.name: float(np.mean([h.revenue for h in self.hosts
                                                         if h.tier == t.name])) for t in cfg.tiers},
                corr_loglam_logprice_comm=float(np.corrcoef(lam_c, rel_p)[0, 1]),
            )
        return out


def simulate(**kw):
    cfg = Config(**kw)
    return Market(cfg).run()


if __name__ == "__main__":
    import json, time
    for pol in POLICIES:
        t0 = time.time()
        m = simulate(policy=pol, seed=1)
        print(pol, round(time.time() - t0, 1), "s",
              json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in m.items() if k not in ("by_class",)}))
