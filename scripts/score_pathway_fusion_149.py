"""Retired command: fail before reading inputs or creating score outputs."""
import sys


def main():
    print('MOGSA_RETIRED: MOGSA and derived parent scoring were removed on 2026-11-04. '
          'Historical outputs are not training labels. No replacement scorer is selected. '
          'See docs/PATHWAY_TARGET_POLICY_20261104.rst.', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
