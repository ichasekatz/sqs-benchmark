"""Compare BCC NbVW SQS method quality across N_RUNS repeats: mcsqs / SCRAPS / Random.

For every bestsqs.out this script:
  1. Reads the structure and computes Warren-Cowley pair-fraction deviation.
  2. Runs corrdump against a shared reference clusters.out / rndstr.in and
     computes the native mcsqs-scale objective (wr/wn/wd weights), making all
     three methods directly comparable on the same scale.
  3. Reads the mcsqs-reported objective from bestcorr.out (mcsqs only) as a
     validation cross-check.
  4. Reads the best relaxed energy/atom from MLIP-relaxed energy files.

Outputs (written to comparison_results/):
  - all_runs.csv              — per-run table with all metrics
  - bcc_method_comparison.png — three-panel box/bar chart
  - bcc_objective_per_run.png — per-run scatter line plot

Run:
    python compare_bcc_methods.py
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

# ------------------------------------------------------------------
# Config — must match the tdb_gen_bcc_*.py scripts
# ------------------------------------------------------------------
n_runs = 10

# Structure label — must match the value set in the driver scripts.
structure_label = "BCC"

path0 = Path("/Users/chasekatz/Desktop/School/Research")
path1 = path0 / "BLADE"
path2 = path0 / "PhaseForge" / "PhaseForge" / "atat" / "data" / "sqsdb"

sqs_base = path0 / "SCRAPS" / "BCC_test_2" / "SQS"
benchmark_base = path1 / "Files" / f"{structure_label}_Benchmark"

out_dir = benchmark_base / "comparison_results"
out_dir.mkdir(parents=True, exist_ok=True)

# NbVW at equimolar composition — matches sqsgen_levels in the driver scripts.
a_ang = 3.165  # representative average lattice parameter, Å
n_atoms = 48  # 4×3×2 supercell × 2 sites/cell (BCC)
pair_cutoff = 3.5  # Å for WC pair-fraction statistics
compositions = {"Nb": 1 / 3, "V": 1 / 3, "W": 1 / 3}
label_map = {"a_A": "Nb", "a_B": "V", "a_C": "W"}

# Reference files from the first mcsqs run — same lattice and cutoffs for all methods.
# corrdump is called with these so all three methods share the same cluster basis.
_ref_dir = path2 / f"{structure_label}mcsqs1_3" / "sqsdb_lev=0_a=0.33333,0.33333,0.33333"
clusters_src = _ref_dir / "clusters.out"
rndstr_src = _ref_dir / "rndstr.in"

# corrdump cutoffs: BCC a≈3.165 Å → 1NN≈2.74, 2NN≈3.16, 3NN≈4.47, 4NN≈5.18, 5NN≈5.48 Å
corrdump_cutoff_flags = ["-2=5.48", "-3=5.18", "-4=4.48"]

# Native mcsqs objective parameters (must match mcsqs_params in tdb_gen_*_mcsqs.py).
mcsqs_wr = 20.0
mcsqs_wn = 0.75
mcsqs_wd = 1.0

# Normalize cluster diameters by first nearest-neighbour distance.
normalize_diameters_by_1nn = True

# Numerical tolerance for treating a correlation as matched.
_match_tol = 1e-10

methods: dict[str, dict] = {
    "mcsqs": {
        "names": [f"{structure_label}mcsqs{i}" for i in range(1, n_runs + 1)],
        "comps": [benchmark_base / f"Comps_mcsqs_run{i}" for i in range(1, n_runs + 1)],
    },
    "SCRAPS": {
        "names": [f"{structure_label}scraps{i}" for i in range(1, n_runs + 1)],
        "comps": [benchmark_base / f"Comps_scraps_run{i}" for i in range(1, n_runs + 1)],
    },
    "Random": {
        "names": [f"{structure_label}random{i}" for i in range(1, n_runs + 1)],
        "comps": [benchmark_base / f"Comps_random_run{i}" for i in range(1, n_runs + 1)],
    },
}


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------
@dataclass(frozen=True)
class _ClusterMeta:
    npoints: int
    diameter: float
    multiplicity: int


@dataclass(frozen=True)
class _CorrRow:
    npoints: int
    diameter: float
    diff: float
    multiplicity: int


# ------------------------------------------------------------------
# Structure helpers
# ------------------------------------------------------------------
def structure_from_bestsqs(path: Path) -> Structure | None:
    """Parse an ATAT bestsqs.out into a pymatgen Structure.

    Args:
        path: Path to bestsqs.out.

    Returns:
        Parsed Structure, or None if parsing fails.
    """
    if not path.exists():
        return None

    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]

    try:
        prim_raw = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(3)])
        sup = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(3, 6)])
    except Exception as exc:
        print(f"  parse error {path}: {exc}")
        return None

    prim = prim_raw * a_ang if abs(prim_raw[0, 0]) < 1.5 else prim_raw
    lat_mat = sup @ prim

    species: list[str] = []
    carts: list[np.ndarray] = []

    for line in lines[6:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            frac = np.array([float(p) for p in parts[:3]])
        except ValueError:
            continue
        cart = frac @ prim
        raw = parts[3]
        elem = label_map.get(raw, raw)
        if "_" in elem or len(elem) > 2 or not elem.isalpha():
            continue
        species.append(elem)
        carts.append(cart)

    return Structure(Lattice(lat_mat), species, carts, coords_are_cartesian=True) if species else None


def pair_fracs(structure: Structure, cutoff: float) -> dict[str, float]:
    """Compute pair-type fractions up to cutoff for a structure.

    Args:
        structure: Pymatgen Structure.
        cutoff: Neighbor cutoff radius in Å.

    Returns:
        Dict mapping "El1-El2" (sorted) to fractional pair count.
    """
    cnt: Counter[str] = Counter()
    for i, site in enumerate(structure):
        for nn in structure.get_neighbors(site, cutoff):
            if nn.index <= i:
                continue
            pair = "-".join(sorted([site.species_string, structure[nn.index].species_string]))
            cnt[pair] += 1
    total = sum(cnt.values())
    return {pair: round(count / total, 5) for pair, count in cnt.items()} if total else {}


def wc_objective(pair_fraction: dict[str, float], comp: dict[str, float]) -> float | None:
    """Warren-Cowley pair-deviation metric relative to ideal random.

    Returns:
        Negative sum of squared pair-fraction deviations from ideal random.
        Closer to 0 = better SQS. None if inputs are empty.
    """
    if not pair_fraction or not comp:
        return None

    total_ideal = sum(
        (xi**2 if i == j else 2 * xi * xj)
        for i, (_, xi) in enumerate(comp.items())
        for j, (_, xj) in enumerate(comp.items())
        if j >= i
    )
    ideal: dict[str, float] = {}
    els = list(comp.keys())
    for i, eli in enumerate(els):
        xi = comp[eli]
        for elj in els[i:]:
            xj = comp[elj]
            pair = "-".join(sorted([eli, elj]))
            raw = xi**2 if eli == elj else 2 * xi * xj
            ideal[pair] = raw / total_ideal

    all_pairs = set(pair_fraction) | set(ideal)
    return -sum((pair_fraction.get(p, 0.0) - ideal.get(p, 0.0)) ** 2 for p in all_pairs)


def relaxed_energy_per_atom(run_comps_dir: Path) -> float | None:
    """Find the lowest relaxed energy/atom in a BLADE Comps output directory.

    Args:
        run_comps_dir: Path to the per-run Comps output directory.

    Returns:
        Best (lowest) energy per atom in eV, or None if no energy files found.
    """
    if not run_comps_dir.exists():
        return None
    best = None
    for ef in run_comps_dir.rglob("energy"):
        try:
            val = float(ef.read_text().strip().split()[0]) / n_atoms
            if best is None or val < best:
                best = val
        except Exception:
            continue
    return best


def find_bestsqs(phase_name: str) -> Path | None:
    """Search sqsdb and staging area for a bestsqs.out for the given phase.

    Args:
        phase_name: Phase identifier, e.g. "BCCmcsqs1".

    Returns:
        Path to the first bestsqs.out found, or None.
    """
    for base in [path2 / f"{phase_name}_3", sqs_base / f"{phase_name}_3"]:
        if not base.exists():
            continue
        for d in sorted(base.glob("sqsdb_lev=*")):
            candidate = d / "bestsqs.out"
            if candidate.exists():
                return candidate
    return None


# ------------------------------------------------------------------
# ATAT / corrdump helpers
# ------------------------------------------------------------------
def _parse_float_stream(text: str) -> list[float]:
    """Parse all floats from whitespace-separated text."""
    vals: list[float] = []
    for line in text.splitlines():
        for tok in line.split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
    return vals


def _parse_clusters_out(path: Path) -> list[_ClusterMeta]:
    """Parse clusters.out into a list of ClusterMeta objects.

    Block format (blank-line separated):
      multiplicity
      diameter
      npoints
      <npoints site lines>

    Args:
        path: Path to clusters.out.

    Returns:
        List of ClusterMeta for all clusters with npoints >= 2.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If no valid clusters are found.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing clusters.out: {path}")

    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    metas: list[_ClusterMeta] = []
    i = 0

    while i < len(lines):
        try:
            mult = int(round(float(lines[i].split()[0])))
            diam = float(lines[i + 1].split()[0])
            npts = int(round(float(lines[i + 2].split()[0])))
        except Exception:
            i += 1
            continue
        if npts >= 2:
            metas.append(_ClusterMeta(npoints=npts, diameter=diam, multiplicity=mult))
        i += 3 + npts

    if not metas:
        raise ValueError(f"No clusters parsed from {path}")

    if normalize_diameters_by_1nn:
        pair_diams = sorted({m.diameter for m in metas if m.npoints == 2 and m.diameter > 1e-12})
        if not pair_diams:
            raise ValueError("Cannot determine 1NN diameter from clusters.out")
        d_1nn = pair_diams[0]
        metas = [_ClusterMeta(npoints=m.npoints, diameter=m.diameter / d_1nn, multiplicity=m.multiplicity) for m in metas]

    return metas


def _corrdump_diffs(bestsqs_path: Path) -> list[float] | None:
    """Run corrdump and return correlation mismatch values (Pi_SQS - Pi_random).

    Args:
        bestsqs_path: Path to bestsqs.out to evaluate.

    Returns:
        List of float mismatch values, or None if corrdump fails.
    """
    if not bestsqs_path.exists() or not clusters_src.exists() or not rndstr_src.exists():
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        shutil.copy(bestsqs_path, tmp / "bestsqs.out")
        shutil.copy(clusters_src, tmp / "clusters.out")
        shutil.copy(rndstr_src, tmp / "rndstr.in")

        cmd = [
            "corrdump",
            *corrdump_cutoff_flags,
            "-ro",
            "-noe",
            "-nop",
            "-cf=clusters.out",
            "-s=bestsqs.out",
            "-l=rndstr.in",
        ]

        try:
            result = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=30, check=False)
        except FileNotFoundError:
            print("  corrdump not found — ensure ATAT binaries are on PATH")
            return None

        if result.returncode != 0:
            print(f"  corrdump failed for {bestsqs_path}: {result.stderr.strip()}")
            return None

        vals = _parse_float_stream(result.stdout)

    return vals if vals else None


def _make_corr_rows(bestsqs_path: Path) -> list[_CorrRow] | None:
    """Combine cluster metadata with corrdump mismatch values.

    Args:
        bestsqs_path: Path to bestsqs.out.

    Returns:
        List of CorrRow objects, or None if either source is unavailable.
    """
    metas = _parse_clusters_out(clusters_src)
    diffs = _corrdump_diffs(bestsqs_path)
    if not diffs:
        return None
    n = min(len(metas), len(diffs))
    if len(metas) != len(diffs):
        print(f"  warning: clusters={len(metas)} diffs={len(diffs)} for {bestsqs_path}; using {n}")
    return [
        _CorrRow(npoints=m.npoints, diameter=m.diameter, diff=d, multiplicity=m.multiplicity)
        for m, d in zip(metas[:n], diffs[:n])
    ]


def corrdump_objective(rows: list[_CorrRow]) -> float:
    """Simple corrdump objective: -sum(diff^2). Closer to 0 = better.

    Args:
        rows: List of CorrRow objects.

    Returns:
        Objective value (non-positive float).
    """
    return -sum(r.diff**2 for r in rows)


def native_mcsqs_objective(
    rows: list[_CorrRow],
    wr: float = mcsqs_wr,
    wn: float = mcsqs_wn,
    wd: float = mcsqs_wd,
) -> float | None:
    """Compute the native mcsqs-scale objective from mismatch rows.

    Reproduces the wr/wn/wd-weighted formula used internally by mcsqs.
    More negative = worse; closer to 0 = better SQS.

    Args:
        rows: List of CorrRow objects.
        wr: Range reward weight (penalises unmatched shells).
        wn: Per-point-order weight decay.
        wd: Diameter decay weight.

    Returns:
        Native-scale objective value, or None if rows is empty.
    """
    if not rows:
        return None

    max_p = max(r.npoints for r in rows)

    smallest_bad_diam: dict[int, float] = {}
    for p in range(2, max_p + 1):
        bad = [r.diameter for r in rows if r.npoints <= p and abs(r.diff) > _match_tol]
        if bad:
            smallest_bad_diam[p] = min(bad)

    if not smallest_bad_diam:
        return 0.0

    weighted_num = 0.0
    weighted_den = 0.0
    for row in rows:
        threshold = smallest_bad_diam.get(row.npoints)
        if threshold is None:
            continue
        if row.diameter + 1e-14 >= threshold:
            w = math.exp(-wd * row.diameter) * (wn ** (row.npoints - 2))
            weighted_num += w * abs(row.diff)
            weighted_den += w

    weighted_error = weighted_num / weighted_den if weighted_den > 0 else 0.0

    range_reward = sum(wr * (wn ** (p - 2)) * d for p, d in smallest_bad_diam.items())

    return weighted_error - range_reward


def read_mcsqs_reported_objective(sqsdb_phase_dir: Path) -> float | None:
    """Read the objective value printed by an actual mcsqs run.

    Args:
        sqsdb_phase_dir: Path to the sqsdb phase directory (e.g. BCCmcsqs1_3).

    Returns:
        Reported objective float, or None if not found.
    """
    if not sqsdb_phase_dir.exists():
        return None

    for d in sorted(sqsdb_phase_dir.glob("sqsdb_lev=*")):
        bestcorr = d / "bestcorr.out"
        if not bestcorr.exists():
            continue
        text = bestcorr.read_text()
        if "Perfect_match" in text:
            return 0.0
        for line in reversed(text.splitlines()):
            if "Objective_function" not in line:
                continue
            cleaned = line.replace("Objective_function", " ").replace("=", " ").replace(":", " ")
            for tok in cleaned.split():
                try:
                    return float(tok)
                except ValueError:
                    pass

    for d in sorted(sqsdb_phase_dir.glob("sqsdb_lev=*")):
        hist = d / "objective_history.txt"
        if not hist.exists():
            continue
        vals = []
        for tok in hist.read_text().split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
        if vals:
            return min(vals)

    return None


# ------------------------------------------------------------------
# Collect data
# ------------------------------------------------------------------
print("Reference files:")
print(f"  clusters.out : {clusters_src}")
print(f"  rndstr.in    : {rndstr_src}")
print(f"\ncorrdump cutoffs : {' '.join(corrdump_cutoff_flags)}")
print(f"native mcsqs     : wr={mcsqs_wr}  wn={mcsqs_wn}  wd={mcsqs_wd}")
print(f"normalize 1NN    : {normalize_diameters_by_1nn}\n")

records: list[dict] = []

for method, cfg in methods.items():
    for i, (phase_name, run_comps_dir) in enumerate(zip(cfg["names"], cfg["comps"]), 1):
        sqsdb_dir = path2 / f"{phase_name}_3"
        bestsqs_file = find_bestsqs(phase_name)

        pair_obj: float | None = None
        if bestsqs_file is not None:
            structure = structure_from_bestsqs(bestsqs_file)
            if structure is not None:
                pair_obj = wc_objective(pair_fracs(structure, pair_cutoff), compositions)

        epa = relaxed_energy_per_atom(run_comps_dir)

        corr_obj: float | None = None
        native_obj: float | None = None
        if bestsqs_file is not None:
            rows = _make_corr_rows(bestsqs_file)
            if rows:
                corr_obj = corrdump_objective(rows)
                native_obj = native_mcsqs_objective(rows)

        reported_obj = read_mcsqs_reported_objective(sqsdb_dir)

        # For mcsqs use the solver-reported objective; for others use our evaluation.
        native_for_comparison = reported_obj if reported_obj is not None else native_obj

        records.append(
            {
                "method": method,
                "run": i,
                "phase_name": phase_name,
                "bestsqs_path": str(bestsqs_file) if bestsqs_file is not None else None,
                "native_scale_objective": native_for_comparison,
                "native_scale_computed": native_obj,
                "mcsqs_reported_objective": reported_obj,
                "corrdump_objective": corr_obj,
                "pair_objective": pair_obj,
                "epa_eV": epa,
            }
        )

        _fmt = lambda v, d: f"{v:.{d}f}" if v is not None else "None"
        print(
            f"  {method:<7} run {i:2d}: "
            f"native={_fmt(native_for_comparison, 6)}  "
            f"computed={_fmt(native_obj, 6)}  "
            f"reported={_fmt(reported_obj, 6)}  "
            f"corrdump={_fmt(corr_obj, 6)}  "
            f"pair={_fmt(pair_obj, 6)}  "
            f"epa={_fmt(epa, 4)}"
        )

df = pd.DataFrame(records)
df.to_csv(out_dir / "all_runs.csv", index=False)
print(f"\nSaved all_runs.csv → {out_dir}")

# ------------------------------------------------------------------
# Stats table
# ------------------------------------------------------------------
print(f"\n{'Method':<12} {'mean':>12} {'std':>12} {'best':>12} {'epa mean':>12}")
print("-" * 64)
print("Native mcsqs-scale objective (more negative = better)\n")

for method in methods:
    sub = df[df["method"] == method]
    obj = sub["native_scale_objective"].dropna()
    epa = sub["epa_eV"].dropna()
    if len(obj):
        print(
            f"  {method:<10} "
            f"{obj.mean():>12.6f} "
            f"{obj.std():>12.6f} "
            f"{obj.min():>12.6f} "
            f"{epa.mean() if len(epa) else float('nan'):>12.4f}"
        )
    else:
        print(f"  {method:<10} no objective data")

# mcsqs validation: computed vs reported
mcsqs_sub = df[df["method"] == "mcsqs"].copy()
if len(mcsqs_sub):
    diffs = (mcsqs_sub["native_scale_computed"] - mcsqs_sub["mcsqs_reported_objective"]).dropna()
    if len(diffs):
        print("\nmcsqs validation (computed − reported):")
        print(f"  mean diff = {diffs.mean():.8f}  max abs = {diffs.abs().max():.8f}")

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
method_order = list(methods.keys())
colors = {"mcsqs": "steelblue", "SCRAPS": "darkorange", "Random": "crimson"}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"BCC NbVW SQS Method Comparison ({n_runs} runs each, ORB MLIP)",
    fontsize=14,
    fontweight="bold",
)

# Panel 1: native-scale objective box plot
ax = axes[0]
obj_data = [df[df["method"] == m]["native_scale_objective"].dropna().tolist() for m in method_order]
bp = ax.boxplot(obj_data, tick_labels=method_order, patch_artist=True, showfliers=True)
for patch, method in zip(bp["boxes"], method_order):
    patch.set_facecolor(colors[method])
    patch.set_alpha(0.7)
flat = [v for group in obj_data for v in group]
if flat:
    pad = (max(flat) - min(flat)) * 0.5 or 0.001
    ax.set_ylim(min(flat) - pad, max(flat) + pad)
ax.set_ylabel("Native mcsqs-scale objective\n(more negative = better)", fontsize=10)
ax.set_title(f"SQS Quality — Native mcsqs Scale\n({n_runs} runs per method)", fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Panel 2: relaxed energy deviation from global mean
ax = axes[1]
e_data = [df[df["method"] == m]["epa_eV"].dropna().tolist() for m in method_order]
if any(e_data):
    all_e = [v for group in e_data for v in group]
    ref_e = float(np.mean(all_e))
    e_dev_mev = [[(v - ref_e) * 1000 for v in group] for group in e_data]
    bp2 = ax.boxplot(e_dev_mev, tick_labels=method_order, patch_artist=True, showfliers=True)
    for patch, method in zip(bp2["boxes"], method_order):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.7)
    ax.axhline(0, color="gray", lw=1, linestyle="--")
    ax.set_ylabel("ΔE from mean (meV/atom)", fontsize=10)
    ax.set_title(f"Relaxed Energy Deviation\n(ref = {ref_e:.4f} eV/atom)", fontsize=11)
    ax.text(0.02, 0.98, f"ref = {ref_e:.4f} eV/atom", transform=ax.transAxes, fontsize=8, va="top", color="gray")
else:
    ax.text(0.5, 0.5, "No energy data", ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
    ax.set_title("Relaxed Energy/Atom", fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Panel 3: mean ± std bar chart
ax = axes[2]
means: list[float] = []
stds: list[float] = []
x_pos: list[int] = []
for k, method in enumerate(method_order):
    sub = df[df["method"] == method]["native_scale_objective"].dropna()
    if len(sub):
        means.append(float(sub.mean()))
        stds.append(float(sub.std()))
        x_pos.append(k)
if means:
    bars = ax.bar(
        x_pos,
        means,
        yerr=stds,
        capsize=5,
        width=0.6,
        color=[colors[method_order[k]] for k in x_pos],
        alpha=0.8,
        edgecolor="white",
        error_kw={"elinewidth": 1.5},
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([method_order[k] for k in x_pos], fontsize=10)
    ax.set_ylabel("Mean native mcsqs-scale objective ± std", fontsize=10)
    ax.set_title(f"Mean SQS Quality\n({n_runs} runs, error = std)", fontsize=11)
    all_obj = [v for k in x_pos for v in df[df["method"] == method_order[k]]["native_scale_objective"].dropna()]
    if all_obj:
        pad = (max(all_obj) - min(all_obj)) * 0.5 or 0.001
        ax.set_ylim(min(all_obj) - pad, max(all_obj) + pad)
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + pad * 0.1,
                f"{mean:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
else:
    ax.text(0.5, 0.5, "No objective data", ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
    ax.set_title("Mean Native mcsqs Objective", fontsize=11)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_png = out_dir / "bcc_method_comparison.png"
plt.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_png}")

# Per-run scatter
fig2, ax2 = plt.subplots(figsize=(12, 5))
for method in method_order:
    sub = df[df["method"] == method].sort_values("run")
    ax2.plot(sub["run"].tolist(), sub["native_scale_objective"].tolist(), "o-", color=colors[method], label=method, alpha=0.8)
all_obj = df["native_scale_objective"].dropna().tolist()
if all_obj:
    pad = (max(all_obj) - min(all_obj)) * 0.3 or 0.001
    ax2.set_ylim(min(all_obj) - pad, max(all_obj) + pad)
ax2.set_xlabel("Run index", fontsize=11)
ax2.set_ylabel("Native mcsqs-scale objective\n(more negative = better)", fontsize=11)
ax2.set_title(f"Native mcsqs-scale Objective per Run (BCC NbVW, {n_runs} repeats)", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
out_png2 = out_dir / "bcc_objective_per_run.png"
fig2.tight_layout()
fig2.savefig(out_png2, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out_png2}")
print(f"CSV:   {out_dir}/all_runs.csv")
