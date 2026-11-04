"""Current RPPA500 input entry. MOGSA scoring and legacy RPPA arguments are retired."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cla_engine.rppa500 import cli

if __name__ == '__main__':
    cli()
