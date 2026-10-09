<div align="center">

# SQS Benchmark

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://opensource.org/license/gpl-3-0)
[![Python](https://img.shields.io/badge/python-3.12-brightgreen.svg)](https://www.python.org/)
[![Lint](https://github.com/ichasekatz/sqs-benchmark/actions/workflows/lint.yml/badge.svg)](https://github.com/ichasekatz/sqs-benchmark/actions/workflows/lint.yml)

**Statistical comparison of SQS generation methods — ATAT mcsqs, SCRAPS, and random assignment — with MLIP relaxation and CALPHAD TDB fitting via BLADE.**

[Report a Bug](https://github.com/ichasekatz/sqs-benchmark/issues/new?labels=bug) · [Request a Feature](https://github.com/ichasekatz/sqs-benchmark/issues/new?labels=enhancement)

</div>

---

## Overview

This benchmark runs three SQS generation methods independently across N repeats on the same alloy system, relaxes every structure with an MLIP via [BLADE](https://github.com/ichasekatz/BLADE), fits CALPHAD TDB databases, and evaluates structural quality on a common set of metrics so all three methods can be compared on the same footing.

| Method | Script | Description |
|---|---|---|
| **mcsqs** | `examples/sqs_bench_mcsqs.py` | ATAT `mcsqs` — simulated-annealing cluster-correlation optimizer |
| **SCRAPS** | `examples/sqs_bench_scraps.py` | SCRAPS cuckoo-search SRO optimizer |
| **Random** | `examples/sqs_bench_random.py` | Random atom assignment — baseline |

Demonstrated on equimolar BCC NbVW (48-atom 4×3×2 supercell, ORB MLIP). Each driver script contains a clearly labelled **Structure definition** block — change `structure_label`, `lattice_a`, `coords`, `phase_list`, and `sqsgen_levels` to target any BLADE-supported prototype.

---

## Dependencies

### Required — BLADE pipeline

The driver scripts import directly from [BLADE](https://github.com/ichasekatz/BLADE) and [SCRAPS](https://github.com/ichasekatz/scraps-perpair). They will not run without these installed in the active Python environment.

| Dependency | Purpose | Install |
|---|---|---|
| `blade` | `BladeCompositions`, `BladeSQS`, `ScrapsSQSGen`, `BladeTDBGen`, `BladeVisualizer` | `pip install -e <BLADE/src>` or `pixi install` (see BLADE repo) |
| `materialsframework` | `GraceCalculator` and MLIP infrastructure used inside BLADE — **[ichasekatz fork](https://github.com/ichasekatz/MaterialsFramework) required** | `pip install -e <PhaseForge/MaterialsFramework>` |
| SCRAPS binary | C++ MPI cuckoo-search optimizer (SCRAPS method only) | `bash build.sh` in `scraps-perpair/` |
| ATAT binaries | `mcsqs`, `corrdump`, `sqs2tdb` on `$PATH` | from ATAT installation |

### Analysis only (pip-installable)

`scripts/compare_methods.py` requires only ATAT's `corrdump` binary plus:

```bash
uv add pymatgen pycalphad matplotlib numpy pandas
```

Analysis can be run on any machine that has the raw `bestsqs.out` results copied over — BLADE and SCRAPS are not needed.

---

## Running

### All methods, 10 runs each

```bash
python scripts/run_all.py
```

### Options

```bash
python scripts/run_all.py --n-runs 5                  # 5 runs each
python scripts/run_all.py --methods mcsqs scraps      # selected methods only
python scripts/run_all.py --start 4                   # resume from run 4
```

Each run injects `run_index` via AST patching (same approach as BLADE's `full_framework.py`) so all runs are independent — unique phase names, unique output directories, independent MLIP relaxations.

### Single run

```bash
python examples/sqs_bench_mcsqs.py
python examples/sqs_bench_scraps.py
python examples/sqs_bench_random.py
```

### Analysis

```bash
python scripts/compare_methods.py
```

Pre-computed 10-run results are in [`results/`](results/).

---

## Metrics

All metrics are computed by `scripts/compare_methods.py`. Closer to 0 = better SQS quality.

---

### Native mcsqs-scale objective

Reproduces the internal objective used by ATAT `mcsqs` from `corrdump` output, making all three methods directly comparable on the same scale.

**Inputs**: cluster-correlation differences Δρ_k, diameter d_k, point-order n_k (2 = pair, 3 = triplet, …), and weights wr, wn, wd.

1. For each point-order *p*, find the smallest cluster diameter *D_p* with |Δρ_k| > ε.

2. Per-cluster weight (applied only to clusters at or beyond the first mismatched shell):

   ```
   w_k = exp(−wd · d_k) · wn^(n_k − 2)
   ```

3. Weighted mean mismatch:

   ```
   ε̄ = Σ_k w_k |Δρ_k| / Σ_k w_k
   ```

4. Range-reward term:

   ```
   R = Σ_p wr · wn^(p−2) · D_p
   ```

5. Objective:

   ```
   f = ε̄ − R
   ```

   More negative = better; perfect SQS → f = 0. Weights used: wr = 20, wn = 0.75, wd = 1.0 (must match `mcsqs_params` in the driver scripts).

For mcsqs runs, `bestcorr.out` provides the solver-reported value as a cross-check. For SCRAPS and Random, the same formula is evaluated from `corrdump` output against the same `clusters.out` / `rndstr.in` reference files from the first mcsqs run.

---

### Corrdump objective

Simple cluster-correlation sum-of-squares from the same `corrdump` output — no range-reward, no weighting:

```
f_cd = −Σ_k (Δρ_k)²
```

---

### Warren-Cowley pair deviation

Measures nearest-neighbour pair fractions against the ideal random alloy without requiring `corrdump`. For equimolar ternary with species *i*, *j* the ideal pair fraction is:

```
f_ij^ideal = 2 · x_i · x_j / Σ_{a≤b} f_ab^raw     (i ≠ j)
f_ii^ideal = x_i²             / …                    (i = j)
```

Observed fractions f_ij^obs are counted within 3.5 Å (first BCC shell). The metric is:

```
f_WC = −Σ_{i≤j} (f_ij^obs − f_ij^ideal)²
```

Computed directly from `bestsqs.out` via pymatgen; no ATAT required.

---

### Relaxed energy per atom

Best (lowest) total MLIP energy per atom across all `sqs_lev=*/energy` files written by BLADE after relaxation. Units: eV/atom.

---

## Results

Pre-computed 10-run results (see [`results/all_runs.csv`](results/all_runs.csv)):

| Method | Native mcsqs-scale (mean ± std) | Corrdump (mean) | WC pair dev (mean) |
|---|---|---|---|
| mcsqs  | −46.2424 ± 0.0003 | −0.0105 | −0.0002 |
| SCRAPS | −46.2388 ± 0.0009 | −0.0250 | −0.0013 |
| Random | −46.2367 ± 0.0009 | −0.0270 | −0.0016 |

mcsqs finds the best structures at 100 s budget with lower variance. SCRAPS and Random produce similar structures at this budget.

---

## Citation

If you use this benchmark, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

## License

Distributed under the GPL-3.0-or-later License. See [LICENSE](LICENSE).

## Contact

Chase Katz — [ichasekatz@tamu.edu](mailto:ichasekatz@tamu.edu)
