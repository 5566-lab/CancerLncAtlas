"""Prepare or reconstruct compact genomic CNV from explicitly bound raw files."""
import argparse
import json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="phase",required=True)
    prepare=commands.add_parser("prepare"); prepare.add_argument("--manifest",required=True)
    prepare.add_argument("--output",required=True); prepare.add_argument("--cancers",nargs="+")
    audit=commands.add_parser("audit"); audit.add_argument("--success",required=True)
    audit.add_argument("--expected-sha256",required=True); audit.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    from pathlib import Path
    from cc_hhgt.v32 import native_genomic_cnv as native
    if args.phase=="prepare":
        result=native.materialize(native.record(Path(args.manifest).resolve()),Path.cwd(),args.output,cancers=args.cancers)
    else:
        path=native.verified_path({"path":args.success,"sha256":args.expected_sha256},Path.cwd())
        output=Path(args.output)
        if output.exists(): raise FileExistsError("Use a new native genomic audit output")
        native.require_host()
        native.write(output.with_name(output.name+".preflight.json"),{"target_host":"149","training_started":False,"input":native.record(path)})
        value=native.validate(path,replay=True)
        result={"format":"NATIVE_GENOMIC_CNV_NUMERICAL_AUDIT_V1","status":"PASS","target_host":"149",
            "training_started":False,"source":native.record(path),"arrays_compared":6*len(value["cancers"]),
            "source_ancestry_complete":False}
        native.write(output,result)
    print(json.dumps(result,indent=2))
    return 0 if result["status"] in {"SUCCESS","PASS"} else 2


if __name__=="__main__": raise SystemExit(main())
