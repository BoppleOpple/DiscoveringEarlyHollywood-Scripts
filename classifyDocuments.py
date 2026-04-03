import argparse
import csv
import json
import math
import os
import random
import re
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from ollama import Client, generate, create, GenerateResponse
from typing import Literal

from utils import read_csv, get_transcripts, match_transcript, is_valid_page

defaultDir = "/mnt/Database Storage/http/capstone"

SYSTEM_PROMPT = """
The user will provide a document describing a film. These documents could be in
any format--i.e. synopsis, sreenplay, script, etc.

Your goal is to catagorize the document into one of the following categories:
- synopsis
- script
- screenplay
- other
- N/A

Reply only with the categorization for the document, e.g. "script".
"""


parser = argparse.ArgumentParser(
    prog="python3 generateMetadata.py",
    description="Generates [count] metadata files for our capstone's dataset",
    epilog="Copyright Liam Hillery, 2025"
)
parser.add_argument(
    "-o",
    "--outfile",
    default="out/classifications.csv",
    type=Path
)
parser.add_argument(
    "-t",
    "--transcript-dir",
    required=False,
    default=f"{defaultDir}/qwen_ocr",
    type=Path,
)
parser.add_argument(
    "-c",
    "--counts-csv",
    required=False,
    default=f"out/page_counts/counts.csv",
    type=Path,
)
parser.add_argument(
    "-i",
    "--id",
    required=False,
    default=None,
    type=str,
)
parser.add_argument(
    "-m",
    "--model",
    required=False,
    default="qwen3:8b",
    type=str,
)
parser.add_argument(
    "-r",
    "--retries",
    required=False,
    default=5,
    type=int,
)
parser.add_argument(
    "--ollama-host",
    required=False,
    default="http://localhost:11434",
    type=str,
)
parser.add_argument(
    "--min-pages",
    required=False,
    default=0,
    type=int,
)
parser.add_argument(
    "--max-pages",
    required=False,
    default=math.inf,
    type=int,
)

# select files based on CLI filters
def select_files(args: argparse.Namespace) -> list[str]:
    files: list[str] = os.listdir(args.transcript_dir)

    print(f"found {len(files)} files and directories")
    # print(files[:10])
    
    valid_pages: list[str] = list(filter(is_valid_page, tqdm(files)))

    print(f"filtered to {len(valid_pages)} transcript files")
    # print(valid_pages[:10])

    ids: tuple[str] = None

    if args.id:
        ids = (args.id,)
    else:
        all_ids = [match_transcript(s).group(1) for s in tqdm(valid_pages)]

        page_counts: dict[str, int] = {}

        for row in read_csv(args.counts_csv):
            page_counts[row[0]] = int(row[1])

        ids = tuple(filter(lambda id: args.min_pages <= page_counts[id] and page_counts[id] <= args.max_pages, tqdm(all_ids)))

    sample: list[str] = list(filter(lambda fname: fname.startswith(ids), tqdm(valid_pages)))

    print(f"final sample: {len(sample)} files")
    # print(sample)

    return sample


def main():
    args = parser.parse_args()
    
    _ollama_client = Client(host=args.ollama_host)
    create(
        model="classificationModel",
        from_=args.model,
        system=SYSTEM_PROMPT
    )

    transcript_files: list[str] = select_files(args)

    # fetch the relevant transcripts
    transcripts: dict[str, list[str]] = get_transcripts(
        args.transcript_dir,
        transcript_files
    )

    # create output directory
    os.makedirs(args.outfile.parent, exist_ok=True)

    classifications: dict = {}


    # import csv if it exists, otherwise create it and its header
    if args.outfile.exists():
        print("importing existing data...")

        for row in tqdm(read_csv(args.outfile)):
            classifications[row[0]] = row[1]
    else:
        # otherwise, create the file with a header row
        print("output file created")
        with open(args.outfile, "w") as f:
            f.write("id,classification\n")

    for id, page_text in tqdm(transcripts.items()):
        # skip if already classified
        if id in classifications:
            continue
        
        # skip if unreadable
        if not page_text:
            print(f"document {id} has no text?")
            continue
        

        for i in range(args.retries):
            failed = False
            response: GenerateResponse = generate(
                model="classificationModel",
                prompt="\n".join(page_text) + "\n\n" + SYSTEM_PROMPT,
                stream=False,
                logprobs=False,
                think=False
            )
            
            # loop again if no response returned
            if not response.response:
                print(f"no response! retrying... ({i+1})")
                continue

            classifications[id] = response.response
            break

        if id in classifications:
            with open(args.outfile, "a") as f:
                f.write(f"{id},{classifications[id]}\n")
        else:
            with open(args.outdir / f"failed_docs.txt", "a") as f:
                print(id, file=f)


if __name__ == "__main__":
    main()
