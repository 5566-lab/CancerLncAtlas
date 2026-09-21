#!/usr/bin/env python3
"""Standardize disease databases and GSE85011 outside the training evidence chain."""

from cc_hhgt.common import configure_logging, load_config, parse_common_args, stage_status
from cc_hhgt.external_validation import standardize_external_evidence
from cc_hhgt.resource_audit import build_resource_usage_manifest


def main() -> None:
    parser = parse_common_args(__doc__)
    args = parser.parse_args()
    configure_logging(args.verbose)
    cfg = load_config(args.config)
    with stage_status(cfg, "01b_standardize_external_validation"):
        result = standardize_external_evidence(cfg)
        build_resource_usage_manifest(cfg)
        print(result)


if __name__ == "__main__":
    main()
