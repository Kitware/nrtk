#! /usr/bin/env python3
"""
Check if opencv-python is installed.
Expects `pip list --format json` as STDIN.
Will exit 0 if present and 1 if not.
"""
import json
import sys

d = json.load(sys.stdin)
has = any(pkg_dict['name'] == 'opencv-python' for pkg_dict in d)
# Reminder: `not` is to invert `True` into a 0 exit code.
exit(int(not has))
