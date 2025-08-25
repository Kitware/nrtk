"""Utility script for synchronizing NRTK's optional dependencies (extras).

This script extracts the [tool.poetry.extras] section from pyproject.toml and writes it
to a YAML file (_extras.yml), which is used at runtime to detect which optional dependencies
are installed. It also prints the status of each extra's dependencies, including their versions.

Can be used as:
  - A standalone CLI tool to generate or update the YAML file
  - A pre-commit hook to ensure extras.yml stays in sync with pyproject.toml
  - An importable module for reporting dependency status inside notebooks or diagnostics

Includes warnings suppression and automatic git staging of updated files when run as a script.

Developed with assistance from AI (ChatGPT and GitHub Copilot).
"""

import importlib
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, TextIO

import pkg_resources
import yaml

import nrtk

_IMPORT_NAME_OVERRIDES = {
    "Pillow": "PIL",
    "scikit-image": "skimage",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
}


# Detect which OpenCV packages are actually installed
def _identify_cv2_package_versions() -> dict[str, str]:
    installed = {}
    for candidate in ("opencv-python", "opencv-python-headless"):
        try:
            dist = pkg_resources.get_distribution(candidate)
            installed[candidate] = dist.version
        except pkg_resources.DistributionNotFound:  # noqa: PERF203
            continue
    return installed


def _try_import(module_name: str) -> tuple[bool, str | None]:
    """Try importing a module and getting its version."""
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "installed (version unknown)")
        return True, version
    except ImportError:
        return False, None


def _get_extras_status(
    extras: dict[str, list[str]],
) -> OrderedDict[str, OrderedDict[str, tuple[bool, str | None]]]:
    """Return a mapping from extras to their dependency status."""
    results: OrderedDict[str, OrderedDict[str, tuple[bool, str | None]]] = OrderedDict()
    reverse_index: defaultdict[str, list[str]] = defaultdict(list)
    cv2_versions = _identify_cv2_package_versions()

    for extra, deps in extras.items():
        extra_status: OrderedDict[str, tuple[bool, str | None]] = OrderedDict()
        for dep in deps:
            import_name = _IMPORT_NAME_OVERRIDES.get(dep, dep.replace("-", "_"))
            ok, ver = _try_import(import_name)
            if dep.startswith("opencv-python"):
                dep_version = cv2_versions.get(dep)
                if not dep_version:
                    ok = False
                    ver = None
            extra_status[dep] = (ok, ver)
            reverse_index[dep].append(extra)
        results[extra] = extra_status

    return results


def print_extras_status(file: TextIO = sys.stdout) -> None:
    import warnings

    print("Detected status of NRTK extras and their dependencies:\n", file=file)
    with warnings.catch_warnings():
        _extras_path: Path = Path(__file__).parent / "_extras.yml"

        with _extras_path.open("r") as f:
            extras: dict[str, list[str]] = yaml.safe_load(f)

        warnings.simplefilter("ignore")
        results = _get_extras_status(extras=extras)
        for extra, deps in results.items():
            print(f"[{extra}]", file=file)
            for dep, (ok, ver) in deps.items():
                status = f"✓ {ver}" if ok else "✗ missing"
                print(f"  - {dep:<25} {status}", file=file)
            print(file=file)

        print("\nFor details about installing NRTK extras, please visit:", file=file, flush=True)
        print(
            f"    https://nrtk.readthedocs.io/en/v{nrtk.__version__}/installation.html#extras\n",
            file=file,
            flush=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract [tool.poetry.extras] from pyproject.toml and write to YAML.")
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml (default: pyproject.toml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/nrtk/utils/_extras.yml"),
        help="Path to output YAML file (default: src/nrtk/utils/_extras.yml)",
    )
    args = parser.parse_args()

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    with args.pyproject.open("rb") as f:
        pyproject: dict[str, Any] = tomllib.load(f)

    extras: dict[str, list[str]] = pyproject.get("tool", {}).get("poetry", {}).get("extras", {})

    with args.output.open("w") as f:
        yaml.dump(extras, f, sort_keys=True)

    print(f"✅ Extras exported to: {args.output.resolve()}")

    import shutil
    import subprocess

    git_path = shutil.which("git")
    if git_path is None:
        print("⚠️  Git not found. Cannot stage extras.yml")
    else:
        try:
            # subprocess call is safe: git path resolved via shutil.which, args.output is validated file path
            subprocess.run([git_path, "add", str(args.output)], check=True)  # noqa: S603
            print(f"✅ Staged: {args.output}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to stage {args.output} (git add error): {e}")
        except OSError as e:
            print(f"⚠️  Failed to stage {args.output} (OS error): {e}")
