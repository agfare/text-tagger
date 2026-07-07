#!/usr/bin/env python3
"""CLI: tag .docx files and write tagged JSON output.

Usage examples:
    # Tag a single file
    python tag_documents.py input.docx --output tagged/input.json

    # Tag all .docx files in a directory
    python tag_documents.py docs/ --output-dir tagged/

    # Use GPU
    python tag_documents.py docs/ --output-dir tagged/ --device 0
"""

import argparse
import sys
from pathlib import Path

from tagger.pipeline import tag_directory, tag_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag .docx files for the DLW recommendation system.")
    parser.add_argument("input", type=Path, help="Path to a .docx file or a directory of .docx files.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (single-file mode).")
    parser.add_argument("--output-dir", type=Path, default=Path("tagged"), help="Output directory (directory mode). Default: tagged/")
    parser.add_argument("--model", default="facebook/bart-large-mnli", help="HuggingFace zero-shot model ID.")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0+ for GPU.")
    args = parser.parse_args()

    if args.input.is_file():
        out = args.output or args.output_dir / args.input.with_suffix(".json").name
        print(f"Tagging {args.input} ...")
        tagged = tag_file(args.input, output_path=out, model_name=args.model, device=args.device)
        print(f"Done. {len(tagged)} chunks tagged → {out}")

    elif args.input.is_dir():
        print(f"Tagging all .docx files under {args.input} ...")
        results = tag_directory(
            args.input,
            output_dir=args.output_dir,
            model_name=args.model,
            device=args.device,
        )
        total = sum(len(v) for v in results.values())
        print(f"Done. {len(results)} files, {total} total chunks → {args.output_dir}/")

    else:
        print(f"Error: {args.input} is not a file or directory.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
