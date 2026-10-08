# GPU Marketplace Simulator

Artifact for the paper *REM: Reliability-Adjusted Matching for Heterogeneous GPU Marketplaces*.

A discrete-event simulator of a GPU-sharing marketplace. It models tiered hosts that fail and recover (exponential, Weibull or seller-correlated), jobs that checkpoint and restart, gang-scheduled multi-node jobs, MIG-style slicing (7 units per GPU), buyer billing for all occupied time, and optionally sellers that adjust prices. It compares matching policies, including REM (reliability-adjusted effective-price matching).

## Requirements

Python 3.10+ with `numpy` and `matplotlib` (see `requirements.txt`). Building the paper needs a TeX distribution with `IEEEtran`.

## Reproduce everything

```bash
pip install -r requirements.txt
python experiments.py        # all experiments -> results/*.json (about 25 min on 2 cores; results are bit-for-bit deterministic)
python plots.py              # figures/*.pdf (Figs. 1-5)
python make_tables.py        # ../paper/generated/*.tex (tables and every number quoted in the paper)
cd ../paper && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

Run a single experiment with `python experiments.py <name>`:

| Name | Paper | What it varies |
| --- | --- | --- |
| `validation` | V1, Table I | single-host job vs closed form |
| `validation_gang` | V2 | 4-host gang job vs closed form |
| `policies` | E1 | matching policy x offered load |
| `sensitivity` | E2 | community reliability x price |
| `checkpoint` | E3 | fixed vs Young/Daly checkpoint interval |
| `mig` | E4 | slicing vs whole-GPU allocation |
| `reputation` | E5 | quality of the failure-rate estimate |
| `sigma` | E6 | host-to-host reliability spread (mean held fixed) |
| `failmodel` | E7 | Weibull shape x seller-correlated outages |
| `gang` | E8 | 4-GPU gang-scheduled jobs |
| `scale` | E9 | 1,000 hosts, 25,000 jobs |
| `pricing` | E10 | fixed vs adaptive seller pricing |

Model extensions are switched on through `Config`: `weibull_k`, `corr_frac` and `group_size`, `JobClass(hosts=...)` for gang jobs, `pricing="adaptive"`, and `ckpt_reputation` to set checkpoint intervals from a different failure-rate signal than placement. With every extension at its default, the simulator reproduces the base model's results exactly.

## Files

| File | Purpose |
| --- | --- |
| `sim.py` | Simulator (failures, gang jobs, slicing, pricing), policies, Daly expected-time model, exact expectation used for validation |
| `experiments.py` | Experiment definitions V1–V2 and E1–E10 |
| `analyze.py` | Means and 95% Student-t confidence intervals across seeds |
| `plots.py` | Figures 1–5 |
| `make_tables.py` | LaTeX tables and number macros, so no result in the paper is typed by hand |
| `results/` | Raw results from the runs reported in the paper |

## Quick use

```python
from sim import simulate
m = simulate(policy="rem", load=0.35, seed=0)
print(m["cost_per_useful_gpu_h"], m["jct_mean"])
```

Policies: `random`, `cheapest`, `verified_cheapest`, `threshold`, `most_reliable`, `rem`, `rem_oracle`.

## Calibration caveat

Tier reliability is calibrated to restart-overhead ranges reported in industry sources (20–40% on unverified marketplace hosts, 3–13% on verified hosts), not to measured host lifetimes. Experiments E2, E6 and E7 sweep reliability, price, host heterogeneity and the failure distribution for this reason. The paper's Discussion section describes a measurement protocol whose outputs (Weibull shape, spread, correlated-outage share) plug directly into `Config`.
