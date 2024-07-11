#! /usr/bin/env python3
"""Prints out the version of opencv-python if installed.

If opencv-python is installed, prints out the version of the package, and will exit code 0 (success).

If opencv-python is not installed, output is undefined and will exit code 1 (failure).

Expects `pip list --format json` as STDIN.
"""
import json
import sys
from typing import Dict, Sequence

d: Sequence[Dict[str, str]] = json.load(sys.stdin)

for pkg_dict in d:
    if pkg_dict["name"] == "opencv-python":
        print(pkg_dict["version"])
        exit(0)
exit(1)
