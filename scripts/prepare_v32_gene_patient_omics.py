"""Build verified gene CNV/promoter methylation matrices on server 149."""
import argparse
from pathlib import Path
from cc_hhgt.v32.gene_patient_omics import materialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(materialize(args.config, args.output, workers=args.workers))


if __name__ == "__main__":
    main()
