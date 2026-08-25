#!/usr/bin/env python3
"""Download reviewed bacterial proteins for putative non-AMP fragments."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
DEFAULT_QUERY = (
    "(reviewed:true) AND (taxonomy_id:2) AND (length:[100 TO 1000]) "
    "AND NOT (keyword:Antimicrobial)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch reviewed bacterial proteins from the official UniProt REST API. "
            "These are parent proteins for length-matched putative-negative fragments."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/uniprot_negative_parents.fasta"),
    )
    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_records < 1:
        raise SystemExit("--max-records must be at least 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    parameters = {"query": args.query, "format": "fasta", "compressed": "false"}
    headers = {"User-Agent": "AMP-Finder-AI/1.0 educational-project"}
    record_count = 0
    saw_fasta_header = False

    with requests.get(
        UNIPROT_STREAM_URL,
        params=parameters,
        headers=headers,
        stream=True,
        timeout=args.timeout,
    ) as response:
        response.raise_for_status()
        with args.output.open("w", encoding="utf-8", newline="\n") as output_handle:
            for line in response.iter_lines(decode_unicode=True):
                if line is None:
                    continue
                if line.startswith(">"):
                    saw_fasta_header = True
                    record_count += 1
                    if record_count > args.max_records:
                        break
                if record_count > 0:
                    output_handle.write(line.rstrip() + "\n")

    if not saw_fasta_header or record_count == 0:
        args.output.unlink(missing_ok=True)
        raise RuntimeError("UniProt response did not contain FASTA records.")

    downloaded_records = min(record_count, args.max_records)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    metadata = {
        "source": "UniProtKB REST API",
        "url": UNIPROT_STREAM_URL,
        "query": args.query,
        "downloaded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "records": downloaded_records,
        "sha256": digest,
        "label_caveat": (
            "These proteins are used only to create putative-negative fragments. "
            "Lack of an antimicrobial annotation is not experimental proof of inactivity."
        ),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {downloaded_records} records to {args.output}")
    print(f"Saved provenance to {metadata_path}")


if __name__ == "__main__":
    main()
