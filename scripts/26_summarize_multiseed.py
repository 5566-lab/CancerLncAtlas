#!/usr/bin/env python3
"""Rebuild calibrated multiseed stability and architecture-ablation reports."""

from cc_hhgt.common import configure_logging, load_config, parse_common_args
from cc_hhgt.multiseed import calibrate_all_seed_models, summarize_multiseed


def main() -> None:
    parser = parse_common_args(__doc__)
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    cfg = load_config(args.config)
    if not args.skip_calibration:
        calibrate_all_seed_models(cfg)
    print(summarize_multiseed(cfg, calibrated=not args.skip_calibration))


if __name__ == "__main__":
    main()
