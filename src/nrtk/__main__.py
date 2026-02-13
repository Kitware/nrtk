"""Command-line interface main."""

import sys

if __name__ == "__main__":
    from nrtk.entrypoints import nrtk_perturber_cli

    sys.exit(nrtk_perturber_cli())
