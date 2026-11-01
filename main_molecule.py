
pass
import argparse
import os
import runpy
import sys
from pathlib import Path
from molecule_test_sampling import BOOTSTRAP_SEEDS, TEST_GROUP_SIZE

ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "BBBP": ROOT / "main_BBBP.py",
    "HIV": ROOT / "main_HIV.py",
    "MUTAG": ROOT / "main_mutag.py",
}

def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected True or False")

def ensure_legacy_data_paths():
    data = ROOT / "data"
    for alias, target in {
        data / "BBBP.csv": data / "bbbp" / "raw" / "BBBP.csv",
        data / "HIV.csv": data / "hiv" / "raw" / "HIV.csv",
    }.items():
        if not target.exists():
            raise FileNotFoundError(f"Missing dataset file: {target}")
        if not alias.exists():
            alias.symlink_to(os.path.relpath(target, alias.parent))
    target = data / "MUTAG"
    alias = data / "MUTAG1" / "MUTAG"
    if not target.exists():
        raise FileNotFoundError(f"Missing dataset directory: {target}")
    if not alias.exists():
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(os.path.relpath(target, alias.parent))

def main():
    p = argparse.ArgumentParser(description="Run one BrainEVO molecule experiment.")
    p.add_argument("--dataset", required=True, choices=sorted(SCRIPTS),
                   help="dataset to run: BBBP, HIV or MUTAG")
    p.add_argument("--bool_bootstrap", type=parse_bool, default=False,
                   help="enable MUTAG bootstrap test sampling")
    args = p.parse_args()
    script = SCRIPTS[args.dataset]
    if not script.exists():
        raise FileNotFoundError(f"Missing experiment script: {script}")
    ensure_legacy_data_paths()
    os.chdir(ROOT)
    print(f"[main_molecule] dataset={args.dataset}")
    print(f"[main_molecule] script={script.name}")
    previous_argv = sys.argv[:]
    sys.argv = [str(script)]
    if args.dataset == "MUTAG":
        sys.argv.extend(["--bool_bootstrap", str(args.bool_bootstrap)])
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv

if __name__ == "__main__":
    main()
