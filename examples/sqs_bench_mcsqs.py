"""BCC NbVW benchmark — SQS generation via ATAT mcsqs.

Single-run driver. Set run_index to distinguish repeated runs.
Output goes to Files/BCC_Benchmark/Comps_mcsqs_run{run_index}/ so all
three method runs can coexist without clobbering each other.

Run standalone:
    python tdb_gen_bcc_mcsqs.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from blade.analysis.blade_visual import BladeVisualizer
from blade.tools.blade_compositions import BladeCompositions
from blade.tools.blade_sqsgen import BladeSQS
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
# phases          : BLADE prototype dict — coords encodes Wyckoff sites;
#                   FCC has 4 sites (0,0,0 + 3 face centres),
#                   HCP needs a≠c and two coords per sublattice
# phase_list      : generator_name = BLADE's CALPHAD generator key,
#                   supercell_size = tile counts (adjust to reach ~48 atoms)
# sqsgen_levels   : equimolar ternary → [1/3, 1/3, 1/3]
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

_phase_key = f"{structure_label}mcsqs{run_index}"

terms_in: dict | None = {
    _phase_key: "1,0:1,0\n2,2:1,0\n",
}
mult_in: dict | None = None
sublattice_map: dict | None = None
sqsgen_in: dict | None = None
fixed_compositions: dict | None = None
system_overrides: dict | None = None
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
        # BCC: 2 sites per unit cell (corner + body-centre), both on sublattice "a"
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

# ------------------------------------------------------------------
# SQS composition levels
# ------------------------------------------------------------------
sqsgen_levels = [
    {"level": 0, "compositions": [[0.33333, 0.33333, 0.33333]], "letter": ["a"]},
]

# ------------------------------------------------------------------
# mcsqs run parameters
# ------------------------------------------------------------------
mcsqs_params = {
    "time": 100,
    "cutoff_mode": "nn",
    "2": 5,
    "3": 4,
    "4": 3,
    "wr": 20,
    "wn": 0.75,
    "wd": 1,
    "parallel_runs": 20,
}

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
# 2. Generate SQS structures
# ------------------------------------------------------------------
if run_sqs:
    for specific_phase in phase_list:
        for len_comp in unique_len_comps:
            lattice = specific_phase["lattice"]
            sqs_gen = BladeSQS(
                phases_dict=phases[lattice],
                sqsgen_levels=sqsgen_levels,
                level=level,
                len_comp=len_comp,
                skip_existing_sqs=skip_existing_sqs,
                sqsgen_in=sqsgen_in.get(lattice) if sqsgen_in else None,
                fixed_compositions=fixed_compositions,
            )
            params = mcsqs_params | {"super_cell_size": specific_phase["supercell_size"]}
            sqs_gen.sqs_gen(phase=specific_phase, paths=paths, params=params)

# ------------------------------------------------------------------
# 3. Fit TDB databases
# ------------------------------------------------------------------
comps_dir = path1 / "Files" / f"{structure_label}_Benchmark" / f"Comps_mcsqs_run{run_index}"

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
        fixed_compositions=fixed_compositions,
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
