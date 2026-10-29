#!/usr/bin/env python3
"""On server 149, prepare measured node features; never train a model."""
import argparse
from cc_hhgt.v32.omics_node_inputs import produce


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(produce(args.config, args.output))


if __name__ == "__main__":
    main()
