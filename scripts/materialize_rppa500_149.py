"""Materialize or validate the pinned official RPPA500 release on host 149."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cla_engine.rppa500 import cli

if __name__ == '__main__':
    cli()
