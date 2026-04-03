import os
import argparse
import zipfile
import random
import re
import json
from pathlib import Path
from PyPDF2 import PdfReader
from tqdm import tqdm

from utils import match_transcript, read_csv

defaultDir = "/mnt/Database Storage/http/capstone"

parser = argparse.ArgumentParser(
    prog="python3 matchFiles.py",
    description="Detects discrepancies between LoC documents at each stage of processing",
    epilog="Copyright Liam Hillery, 2025"
)

parser.add_argument(
    "-d",
    "--document-dir",
    default=Path(f"{defaultDir}/film_copyright"),
    type=Path
)

parser.add_argument(
    "-t",
    "--transcript-dir",
    default=Path(f"{defaultDir}/qwen_ocr"),
    type=Path
)
parser.add_argument(
    "-m",
    "--metadata-dir",
    default=Path(f"{defaultDir}/cleaned_copyright_with_metadata"),
    type=Path
)
parser.add_argument(
    "-c",
    "--counts-csv",
    required=False,
    default=f"out/page_counts/counts.csv",
    type=Path,
)
parser.add_argument(
    "-o",
    "--outdir",
    default=Path("./out"),
    type=Path
)


def main():

    args = parser.parse_args()

    documents = set(os.listdir(args.document_dir))

    transcripts: dict[str, list[int]] = {}
    for fname in os.listdir(args.transcript_dir):
        match: re.Match = match_transcript(fname)

        if match is None:
            continue

        id: str = match.group(1)
        page: int = int(match.group(2))

        if id not in transcripts:
            transcripts[id] = []

        transcripts[id].append(page)

    transcript_ids: set = set(transcripts.keys())

    page_counts: dict[str, int] = {}
    for row in read_csv(args.counts_csv):
        page_counts[row[0]] = int(row[1])


    counts: dict[str, list[str]] = {
        "failed_transcripts": None,
        "missing_pages": None,
    }

    difference = documents.difference(transcripts)

    print(f"{len(difference)} files in documents not in transcripts")
    # print(list(difference))

    counts["failed_transcripts"] = list(difference)

    missing_pages: list[str] = list(filter(lambda id: len(transcripts[id]) < page_counts[id], transcript_ids))
    print(f"{len(missing_pages)} trasncripts missing pages")

    counts["missing_pages"] = missing_pages

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    with open(f"{args.outdir}/missing.json", "w") as tallies_json:
        json.dump(counts, tallies_json)



if __name__ == "__main__":
    main()
