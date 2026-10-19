"""Query GDC STAR metadata or assemble verified downloads on host 149."""
import argparse
from pathlib import Path
from cc_hhgt.v32.gdc_star_counts import query, materialize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    metadata = commands.add_parser("query")
    metadata.add_argument("--cancers", required=True, help="Comma-separated TCGA cancer abbreviations")
    metadata.add_argument("--output", required=True, type=Path)
    assembly = commands.add_parser("materialize")
    for name in ("manifest", "download-root", "output"): assembly.add_argument("--" + name, required=True, type=Path)
    assembly.add_argument("--cancer", required=True)
    assembly.add_argument("--download", action="store_true", help="Fetch missing open files; existing files must pass GDC size/MD5")
    args = parser.parse_args(argv)
    if args.command == "query": result = query(args.cancers.split(","), args.output)
    else: result = materialize(args.manifest, args.cancer, args.download_root, args.output, download=args.download)
    print(result)
    return 0


if __name__ == "__main__": raise SystemExit(main())
