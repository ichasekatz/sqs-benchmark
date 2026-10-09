"""Run all three SQS benchmark methods for N_RUNS repeats each.

Executes sqs_bench_mcsqs.py, sqs_bench_scraps.py, and
sqs_bench_random.py in sequence for run indices 1 through N_RUNS,
injecting the run_index variable via AST patching (same approach as
BLADE's full_framework.py). Each run produces an independent SQS
structure (unique phase name, unique output directory) so results
can be compared statistically by compare_methods.py.

Usage:
    python run_all.py                 # all 3 methods, 10 runs each
    python run_all.py --n-runs 5      # 5 runs each
    python run_all.py --methods mcsqs scraps   # only selected methods
    python run_all.py --start 3       # resume from run 3
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Which methods to run and how many repeats
# ------------------------------------------------------------------
_method_scripts: dict[str, str] = {
    "mcsqs": "sqs_bench_mcsqs.py",
    "scraps": "sqs_bench_scraps.py",
    "random": "sqs_bench_random.py",
}


def _run_script_with_index(script_path: Path, run_index: int) -> None:
    """Execute a sqs_bench_*.py script with a specific run_index injected.

    Uses AST patching to replace the run_index top-level assignment before
    execution, matching the full_framework.py approach in BLADE examples.

    Args:
        script_path: Absolute path to the driver script.
        run_index: Integer run number to inject as run_index = <value>.
    """
    source = script_path.read_text()
    tree = ast.parse(source, filename=str(script_path))

    # Remove any existing run_index and random_seed assignments so the
    # injected values take effect without duplication.
    _drop = {"run_index", "random_seed"}
    tree.body = [node for node in tree.body if not _is_assignment_to(node, _drop)]
    ast.fix_missing_locations(tree)

    # Derive random_seed from run_index for reproducible random structures.
    namespace: dict = {
        "__name__": "__main__",
        "__file__": str(script_path),
        "__package__": None,
        "run_index": run_index,
        "random_seed": run_index * 100,
    }

    print(f"\n{'=' * 60}")
    print(f"  {script_path.name}  run_index={run_index}")
    print(f"{'=' * 60}\n")

    exec(compile(tree, str(script_path), "exec"), namespace)  # noqa: S102


def _is_assignment_to(node: ast.stmt, names: set[str]) -> bool:
    """Return True when node is a simple assignment to any name in names."""
    if isinstance(node, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id in names
    return False


def main() -> int:
    """Parse CLI arguments and run all configured benchmark repeats."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of independent repeats per method (default: 10).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="First run index to execute — use to resume a partial run (default: 1).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(_method_scripts),
        default=list(_method_scripts),
        help="Subset of methods to run (default: all three).",
    )
    args = parser.parse_args()

    here = Path(__file__).parent
    examples_dir = here.parent / "examples"

    for method in args.methods:
        script = examples_dir / _method_scripts[method]
        if not script.exists():
            print(f"ERROR: missing script {script}", file=sys.stderr)
            return 1
        for run_index in range(args.start, args.n_runs + 1):
            _run_script_with_index(script, run_index)

    print("\nAll benchmark runs complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
