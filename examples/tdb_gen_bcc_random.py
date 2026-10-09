"""BCC NbVW benchmark — random SQS structures → TDB fitting.

Mirrors tdb_gen_bcc_mcsqs.py, replacing BladeSQS with randomly generated
structures. Atoms are randomly assigned to M-sites according to target
fractions and written as ATAT-format bestsqs.out directly into the sqsdb.
BladeTDBGen then handles MLIP relaxation and fitting normally.

Output goes to Files/BCC_Benchmark/Comps_random_run{run_index}/ so all
three method runs can coexist without clobbering each other.

Run standalone:
    python tdb_gen_bcc_random.py
"""

from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from blade.analysis.blade_visual import BladeVisualizer
from blade.tools.blade_compositions import BladeCompositions
from blade.tools.blade_tdb_gen import BladeTDBGen
from pycalphad import Database

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
path0 = Path("/Users/chasekatz/Desktop/School/Research")
path1 = path0 / "BLADE"
path2 = path0 / "PhaseForge" / "PhaseForge" / "atat" / "data" / "sqsdb"
paths = [path0, path1, path2]

# ------------------------------------------------------------------
# Run index — increment per repeat to keep phase names unique
# ------------------------------------------------------------------
run_index = 1

# ------------------------------------------------------------------
# Run flags
# ------------------------------------------------------------------
level = 0
run_sqs = True
skip_existing_sqs = False

run_tdb = True
skip_existing_tdb = False
refit_existing_tdb = False
skip_existing_plots = False
generate_gibbs_energy = True
generate_gibbs_mixing = True
generate_phase_diagram = True
generate_combined_phase_diagram = True
generate_contcar_plots = True

# Each independent run gets a distinct seed so its random structure differs.
# Set to None for truly random on every execution.
random_seed: int | None = run_index * 100

# ------------------------------------------------------------------
# MLIP calculator
# ------------------------------------------------------------------
mlip = "orb"
mlip_kwargs = {"steps": 1000, "device": "cpu"}

tdb_params = {
    "fmax": 1e-4,
    "verbose": True,
    "calculator": mlip,
    "calculator_kwargs": mlip_kwargs,
    "t_min": 298.15,
    "t_max": 10000.0,
    "sro": False,
    "bv": 1e-3,
    "phonon": False,
    "open_calphad": False,
    "track_trajectory": True,
    "terms": None,
}

# ------------------------------------------------------------------
# Structure definition — change these blocks for other lattice types.
#
# structure_label : short tag written into phase keys and output dirs
# primary_elements: metals to include; lattice_a must cover each one
# lattice_a       : element → equilibrium lattice parameter (Å)
# phases          : BLADE prototype dict — coords encodes Wyckoff sites
# phase_list      : generator_name = BLADE's CALPHAD generator key,
#                   supercell_size = tile counts (adjust for ~48 atoms)
# sqsgen_levels   : equimolar ternary → [1/3, 1/3, 1/3]
# prim_sites      : fractional coords + "M" for each variable site;
#                   used by _write_bestsqs to tile the supercell
# ------------------------------------------------------------------
structure_label = "BCC"

# ------------------------------------------------------------------
# Elements and composition constraints
# ------------------------------------------------------------------
primary_elements = ["V", "W", "Nb"]
secondary_elements: list[str] = []

primary_min = 3
primary_max = 3
secondary_min = 0
secondary_max = 0

# ------------------------------------------------------------------
# Lattice constants (Å)
# ------------------------------------------------------------------
lattice_a = {"V": 3.030, "W": 3.165, "Nb": 3.300}

_active = [el for el in primary_elements if el in lattice_a]
_avg_a = sum(lattice_a[el] for el in _active) / len(_active)
print(f"{structure_label} lattice estimate: a={_avg_a:.4f} Å  (avg of {_active})")

_phase_key = f"{structure_label}random{run_index}"

terms_in: dict | None = {
    _phase_key: "1,0:1,0\n2,2:1,0\n",
}
mult_in: dict | None = None
sublattice_map: dict | None = None
run_movie = False

# ------------------------------------------------------------------
# Phase prototype
# ------------------------------------------------------------------
phases: dict[str, dict] = {
    _phase_key: {
        "a": _avg_a,
        "b": _avg_a,
        "c": _avg_a,
        "alpha": 90,
        "beta": 90,
        "gamma": 90,
        "vectors": "1 0 0\n0 1 0\n0 0 1\n",
        # BCC: 2 sites (corner + body-centre), both sublattice "a"
        # FCC: 4 sites — "0 0 0 a\n0.5 0.5 0 a\n0 0.5 0.5 a\n0.5 0 0.5 a\n"
        # HCP: 2 sites — set b=a, c=a*1.633, alpha/beta=90, gamma=120
        "coords": ("0.000000 0.000000 0.000000 a\n0.500000 0.500000 0.500000 a\n"),
    },
}

phase_list = [
    {
        "generator_name": structure_label,
        "lattice": _phase_key,
        # 4×3×2 × 2 sites = 48 atoms for BCC; adjust for other structures
        "supercell_size": (4, 3, 2),
    },
]

liquid = False

sqsgen_levels = [
    {"level": 0, "compositions": [[0.33333, 0.33333, 0.33333]], "letter": ["a"]},
]

# ------------------------------------------------------------------
# Random structure writer
# ------------------------------------------------------------------

# Primitive M-sites tiled into the supercell; one entry per site in the unit cell.
# BCC: 2 sites.  FCC: 4 sites.  HCP: 2 sites.
_prim_sites: list[tuple[float, float, float, str]] = [
    (0.0, 0.0, 0.0, "M"),
    (0.5, 0.5, 0.5, "M"),
]

# ATAT generic labels — uppercase letters A, B, … become a_A, a_B, …
_atat_abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _write_bestsqs(
    elements: list[str],
    fracs: list[float],
    supercell_size: tuple[int, int, int],
    a: float,
    dest: Path,
    seed: int | None,
) -> None:
    """Write ATAT-format bestsqs.out and rndstr.in into dest for a BCC supercell.

    Uses generic labels (a_A, a_B, …) for M-sites. ATAT substitutes these
    with actual element symbols from species.in during sqs2tdb. Uppercase
    labels are copied verbatim; lowercase labels are substituted.

    Args:
        elements: Element symbols in the same order as fracs.
        fracs: Target site fractions, one per element.
        supercell_size: (nx, ny, nz) multiples of the primitive cell.
        a: BCC lattice parameter in Å.
        dest: Directory to write into (created if absent).
        seed: RNG seed for reproducible random assignment; None = truly random.
    """
    dest.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = supercell_size
    prim = np.diag([a, a, a])

    # Build the M-site element list by randomly assigning according to fracs.
    n_m_per_cell = sum(1 for *_, t in _prim_sites if t == "M")
    n_m = nx * ny * nz * n_m_per_cell

    counts = [max(0, round(f * n_m)) for f in fracs]
    diff = n_m - sum(counts)
    for i in range(abs(diff)):
        counts[i % len(counts)] += 1 if diff > 0 else -1
        counts[i % len(counts)] = max(0, counts[i % len(counts)])

    m_labels: list[str] = []
    for idx, cnt in enumerate(counts):
        m_labels.extend([f"a_{_atat_abc[idx]}"] * cnt)

    rng = random.Random(seed)
    rng.shuffle(m_labels)

    # Primitive lattice vectors (rows), then supercell multiples, then atoms.
    lines: list[str] = []
    for row in prim:
        lines.append(" ".join(f"{v:.10f}" for v in row))
    lines.append(f"{float(nx):.1f} 0.0 0.0")
    lines.append(f"0.0 {float(ny):.1f} 0.0")
    lines.append(f"0.0 0.0 {float(nz):.1f}")

    m_idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for fx, fy, fz, site_type in _prim_sites:
                    label = m_labels[m_idx] if site_type == "M" else site_type
                    if site_type == "M":
                        m_idx += 1
                    lines.append(f"{fx + ix:.10f} {fy + iy:.10f} {fz + iz:.10f} {label}")

    (dest / "bestsqs.out").write_text("\n".join(lines) + "\n")
    (dest / "wait").touch()

    active = [(f"{_atat_abc[i]}", fracs[i]) for i in range(len(fracs)) if fracs[i] > 0]
    m_occ = ",".join(f"a_{letter}={frac}" for letter, frac in active)
    rndstr_lines = [
        "",
        f"{a:.6f} {a:.6f} {a:.6f} 90 90 90",
        "1 0 0",
        "0 1 0",
        "0 0 1",
        "",
        f"0.000000 0.000000 0.000000 {m_occ}",
        f"0.500000 0.500000 0.500000 {m_occ}",
    ]
    (dest / "rndstr.in").write_text("\n".join(rndstr_lines) + "\n")

    comp_str = "".join(f"{el}{cnt}" for el, cnt in zip(elements, counts) if cnt > 0)
    print(f"  random structure → {dest.name} ({comp_str})")


# ------------------------------------------------------------------
# 1. Generate compositions
# ------------------------------------------------------------------
composer = BladeCompositions(
    primary_elements=primary_elements,
    secondary_elements=secondary_elements,
    primary_min=primary_min,
    primary_max=primary_max,
    secondary_min=secondary_min,
    secondary_max=secondary_max,
)

composition_list = composer.generate_compositions()
unique_len_comps = composer.get_systems()

print(f"Compositions ({len(composition_list)} total): {composition_list}")
print(f"System sizes: {unique_len_comps}")

# ------------------------------------------------------------------
# 2. Generate random structures and write to sqsdb
# ------------------------------------------------------------------
if run_sqs:
    rng_seed = random_seed

    for specific_phase in phase_list:
        lattice = specific_phase["lattice"]
        sc_size = specific_phase["supercell_size"]
        a_param = phases[lattice]["a"]

        for len_comp in unique_len_comps:
            sqs_dir = path1 / "Files" / "SQS" / f"{lattice}_{len_comp}"

            if skip_existing_sqs and sqs_dir.exists():
                print(f"Skipping existing SQS dir: {sqs_dir}")
                continue

            if sqs_dir.exists():
                shutil.rmtree(sqs_dir)
            sqs_dir.mkdir(parents=True, exist_ok=True)

            # Minimal rndstr.skel required by BladeTDBGen directory scanning.
            rndstr_skel = (
                f"{a_param} {a_param} {a_param} 90 90 90\n"
                "1 0 0\n0 1 0\n0 0 1\n"
                "0.000000 0.000000 0.000000 a\n"
                "0.500000 0.500000 0.500000 a\n"
            )
            (sqs_dir / "rndstr.skel").write_text(rndstr_skel)

            written: set[tuple[float, ...]] = set()

            for lvl_def in sqsgen_levels:
                if lvl_def["level"] > level:
                    continue
                for comp_fracs in lvl_def["compositions"]:
                    for comp_entry in [c for c in composition_list if len(c) == len_comp]:
                        els = list(comp_entry)
                        raw = [float(f) for f in comp_fracs]
                        raw += [0.0] * max(0, len_comp - len(raw))
                        raw = raw[:len_comp]
                        frac_map = dict(zip(els, raw))
                        fracs = tuple(frac_map[e] for e in els)

                        if fracs in written:
                            break
                        written.add(fracs)

                        frac_strs = [str(round(frac_map[e], 6)) for e in els]
                        while len(frac_strs) > 1 and frac_strs[-1] == "0.0":
                            frac_strs.pop()
                        dir_name = f"sqsdb_lev={lvl_def['level']}_a=" + ",".join(frac_strs)

                        _write_bestsqs(
                            elements=els,
                            fracs=list(fracs),
                            supercell_size=sc_size,
                            a=a_param,
                            dest=sqs_dir / dir_name,
                            seed=rng_seed,
                        )

                        if rng_seed is not None:
                            rng_seed += 1
                        break  # one entry per composition

            # sqsgen.in — ATAT uses this to discover composition directory names.
            sqsgen_lines: list[str] = []
            for lvl_def in sqsgen_levels:
                if lvl_def["level"] > level:
                    continue
                for comp_fracs in lvl_def["compositions"]:
                    frac_strs = [str(f) for f in comp_fracs]
                    while len(frac_strs) > 1 and frac_strs[-1] == "0.0":
                        frac_strs.pop()
                    sqsgen_lines.append(f"level={lvl_def['level']}\t\ta=" + ",".join(frac_strs))
            (sqs_dir / "sqsgen.in").write_text("\n".join(sqsgen_lines) + "\n")

            print(f"Written {len(written)} random structures to {sqs_dir}")

# ------------------------------------------------------------------
# 3. Fit TDB databases
# ------------------------------------------------------------------
comps_dir = path1 / "Files" / f"{structure_label}_Benchmark" / f"Comps_random_run{run_index}"

if run_tdb:
    gen = BladeTDBGen(
        phases=phase_list,
        phases_dict=phases,
        liquid=liquid,
        paths=paths,
        composition_list=composition_list,
        level=level,
        skip_existing=skip_existing_tdb,
        refit_existing=refit_existing_tdb,
        output_dir=comps_dir,
        terms_in=terms_in,
        mult_in=mult_in,
        sublattice_map=sublattice_map,
        tdb_params=tdb_params,
    )
    gen.fit()

# ------------------------------------------------------------------
# 4. Plot Gibbs energy and phase diagrams
# ------------------------------------------------------------------
_coords = phases[phase_list[0]["lattice"]]["coords"]
_labels = [ln.strip().split()[-1] for ln in _coords.strip().splitlines() if ln.strip()]
_fixed = [label for label in _labels if not (len(label) == 1 and label.islower())]
remove_elements = set(_fixed)
fixed_species = {el: count / len(_labels) for el, count in Counter(_fixed).items()}

filt_comp_list = [[el for el in comp if el not in remove_elements] for comp in composition_list]

viz = BladeVisualizer()

for _comp, comp_filt in zip(composition_list, filt_comp_list):
    if len(comp_filt) != 2:
        continue
    comp_name = "".join(comp_filt)
    comp_dir = comps_dir / comp_name
    if not comp_dir.exists():
        continue
    phase_name = f"{phase_list[0]['generator_name']}1_{len(comp_filt)}"
    tdb_phases = [phase_name]
    plot_paths = (
        comp_dir / f"{comp_name}_Gibbs_Energy.png",
        comp_dir / f"{comp_name}_Gibbs_Mixing.png",
        comp_dir / f"{comp_name}_Phase_Diagram.png",
    )
    make_energy = generate_gibbs_energy and not (skip_existing_plots and plot_paths[0].exists())
    make_mixing = generate_gibbs_mixing and not (skip_existing_plots and plot_paths[1].exists())
    make_phase = generate_phase_diagram and not (skip_existing_plots and plot_paths[2].exists())
    if not any((make_energy, make_mixing, make_phase)):
        continue
    for tdb_file in comp_dir.glob("*.tdb"):
        tdb = Database(str(tdb_file))
        if make_energy:
            viz.plot_gibbs_energy(
                tdb=tdb,
                metals=comp_filt,
                phase=phase_name,
                fixed_species=fixed_species,
                temperatures=[300, 1000, 2000, 3000, 4000],
                output_path=plot_paths[0],
            )
        if make_mixing:
            viz.plot_gibbs_mixing(
                tdb=tdb,
                metals=comp_filt,
                phase=phase_name,
                fixed_species=fixed_species,
                temperatures=[300, 1000, 2000, 3000, 4000],
                output_path=plot_paths[1],
            )
        if make_phase:
            viz.plot_binary_phase_diagram(
                tdb=tdb,
                metals=comp_filt,
                phases=tdb_phases,
                fixed_species=fixed_species,
                temperature_range=(300, 4500, 50),
                output_path=plot_paths[2],
            )

# ------------------------------------------------------------------
# 5. Visualize
# ------------------------------------------------------------------
if generate_combined_phase_diagram:
    pngs = []
    for comp_filt in filt_comp_list:
        comp_dir = comps_dir / "".join(comp_filt)
        pngs.extend(comp_dir.glob("*_Phase_Diagram.png"))
    out_pd = comps_dir / "Combined_Phase_Diagrams.png"
    if pngs and not (skip_existing_plots and out_pd.exists()):
        viz.phase_diagram(pngs, save=out_pd)
        print(f"Combined phase diagrams → {out_pd}")

if generate_contcar_plots:
    for comp_filt in filt_comp_list:
        comp_name = "".join(comp_filt)
        comp_dir = comps_dir / comp_name
        if not comp_dir.exists():
            continue
        for phase_dir in sorted(p for p in comp_dir.iterdir() if p.is_dir()):
            contcars = sorted(phase_dir.glob("sqs_lev=*/CONTCAR"))
            if not contcars:
                continue
            out = comp_dir / f"Combined_CONTCARs_{comp_name}_{phase_dir.name}.png"
            if skip_existing_plots and out.exists():
                continue
            viz.contcar(contcars, save=out)
            print(f"Saved combined CONTCARs → {out}")

if run_movie:
    viz.make_combined_relaxation_movie(
        composition_list=filt_comp_list,
        path1=comps_dir,
        traj_name="relaxation_live.xyz",
        fps=10,
    )
