#!/usr/bin/env python3
"""Register actual selected/context releases for the existing website; no deployment.

CURRENT_ANALYSIS_WEB_INPUT_V2: release_id, selected_analysis={path,sha256},
components containing state/clinical/drug/evidence/external/single_cell/patient. Each has either
status=AVAILABLE and binding={path,sha256}, or an explicit NOT_PRODUCED,
SOURCE_UNAVAILABLE or NOT_APPLICABLE status with reason. No legacy fallback.
"""
import argparse
import json
from cc_hhgt.v32.current_analysis_web import materialize_current_analysis_web


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True); parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(materialize_current_analysis_web(args.manifest,args.output)))
    return 0


if __name__=="__main__": raise SystemExit(main())
