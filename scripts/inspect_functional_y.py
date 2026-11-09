"""Print verified counts for one cancer and independent label context."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cla_engine.functional_y import load_functional_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--cancer', required=True)
    parser.add_argument('--context', default='sealed_test')
    args = parser.parse_args()
    result = load_functional_context(args.catalog, args.cancer, args.context)
    print(json.dumps(result.counts(), indent=2))


if __name__ == '__main__':
    main()
