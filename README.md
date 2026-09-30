# GatedBDRIS

Simulation code for the article

**Gated Hybrid Active and Passive BD-RIS for Finite Blocklength RSMA ISAC With a Matrix CRB Requirement**
P. Sudharshan Babu, O. Gottam, S. Kota and P. Dhilleswararao (submitted to the IEEE Open Journal of the Communications Society).

The code reproduces every number and table of the article: a multiuser rate splitting (RSMA) ISAC downlink with short packets, assisted by a beyond diagonal RIS whose amplifier equipped ports are biased per channel under one power budget, with a nuisance aware matrix Cramér Rao bound on angle and range.

## Contents
| File | Purpose |
|---|---|
| `src/system_model.py` | Geometry, channels, hybrid active and passive BD-RIS scattering, finite blocklength rates, exact 4x4 Fisher information and matrix CRB, gate retraction |
| `src/algorithm.py` | Gated penalty manifold (GPM) method (Algorithm 1): convex program (P4) in CVXPY/CLARABEL, closed form manifold projections, dual priced switching sweep, baselines (SDMA, NOMA, passive, fully active, diagonal, fixed partition, scalar CRB, Shannon reference) |
| `src/runner.py` | Base Monte Carlo campaign (blocklength, power, CRB threshold, partition and exhaustive sweep, QoS, range requirement, convergence, stationarity rate, CDF) |
| `src/runner_ext.py` | Ablation, scaling in M and K, device sensitivity (noise figure, bias power, off state loss), CSI error model, passive echo, fixed surface access, random initialisations |
| `src/verify.py` | Numerical checks of the lemmas and propositions (V1 to V9) |
| `src/make_figs.py`, `src/make_figs_ext.py`, `src/plots.py` | Aggregation into `results/summary.json` and quick look figures |
| `src/make_facts.py`, `src/make_facts_ext.py` | Every number quoted in the article, with bootstrap confidence intervals, written to `tex/facts.tex` |
| `run_all.sh` | One shot pipeline |

## Requirements
Python 3.10 or later and the packages in `requirements.txt` (NumPy, SciPy, CVXPY with CLARABEL, Matplotlib).

```bash
pip install -r requirements.txt
```

## Reproduce
```bash
bash run_all.sh
```
Default sizes are those of the article: 50 channel realisations per point for the main sweeps, 100 for the partition study, 30 for the side studies and 100 designs for the reliability study. They can be changed through the environment variables `NREAL`, `NPART`, `NHI`, `NCDF` and `WORKERS`, e.g. `NREAL=10 NHI=5 NCDF=10 NPART=10 bash run_all.sh` for a quick run. Jobs are checkpointed in append only JSONL files under `results/` and resume on relaunch. Confidence intervals are 95 percent percentile bootstrap intervals (4000 paired resamples, fixed seed).

Individual studies: `cd src && python3 runner.py 4` (partition study) or `python3 runner_ext.py a` (ablation); see the docstrings for the study keys.

## Citation
Please cite the article if you use this code (full reference to be added on publication).
