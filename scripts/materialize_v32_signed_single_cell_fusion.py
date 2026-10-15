#!/usr/bin/env python3
"""Bind the current signed SC-HHGT donor/context predictions to fusion."""
import argparse
import json
from pathlib import Path

from cc_hhgt.v32.single_cell_signed_fusion import materialize_signed_sc_fusion


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-pointer",type=Path,required=True)
    parser.add_argument("--candidates",type=Path,required=True)
    parser.add_argument("--donor-crosswalk",type=Path,required=True,
                        help="Parquet dataset_id/query_donor_id/canonical_donor_id; reconcile repeated patients across studies")
    parser.add_argument("--min-donors",type=int,default=3)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    result=materialize_signed_sc_fusion(args.current_pointer,args.candidates,args.donor_crosswalk,args.output,
                                      min_donors=args.min_donors)
    print(json.dumps(result,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
