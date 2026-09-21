#!/usr/bin/env python3
"""Evaluate frozen all-candidate rankings against isolated external evidence."""

from cc_hhgt.common import configure_logging, load_config, parse_common_args, stage_status
from cc_hhgt.external_validation import run_external_validation
from cc_hhgt.resource_audit import refresh_resource_usage_after_model


def main() -> None:
    parser = parse_common_args(__doc__)
    args = parser.parse_args()
    configure_logging(args.verbose)
    cfg = load_config(args.config)
    with stage_status(cfg, "23_run_external_validation"):
        result = run_external_validation(cfg)
        refresh_resource_usage_after_model(cfg)
        print(result)


if __name__ == "__main__":
    main()
