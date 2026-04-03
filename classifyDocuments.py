import argparse
import csv
import json
import math
import numpy as np
import os
import random
import re
from matplotlib import plt
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
- other

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

    # get the total number of documents in each category
    category_dist = np.unique_counts(list(classifications.values()))

    # plot them in a pie chart
    proportions_fig: plt.Figure = plt.figure()
    plt.pie(category_dist.counts, labels=category_dist.values)
    proportions_fig.show()

    # create a mapping from page count to all ids with that number of pages
    map_pages_to_id: dict[int, list[str]] = {}

    for row in read_csv(args.counts_csv):
        id: str = row[0]
        count: int = int(row[1])

        # create a list for the given count and add the id
        if count not in map_pages_to_id:
            map_pages_to_id[count] = []

        map_pages_to_id[count].append(id)

    # create a mapping from page count to category tallies
    map_pages_to_classifications: dict[int, dict[str, int]] = {}

    for page_count, ids in map_pages_to_id.items():
        group_classifications: list = []
        for id in ids:
            if id in classifications:
                group_classifications.append(classifications[id])
            else:
                print(f"could not find id {id}")
        dist = np.unique_counts(group_classifications)

        map_pages_to_classifications[page_count] = {}
        for i in range(len(dist.values)):
            map_pages_to_classifications[page_count][str(dist.values[i])] = dist.counts[i] / len(ids)

    # pprint(map_pages_to_classifications)

    x: np.ndarray = np.array(list(map_pages_to_id.keys()))
    synopsis_bars: np.ndarray = np.array(
        [
            (
                map_pages_to_classifications[count]["synopsis"]
                if "synopsis" in map_pages_to_classifications[count]
                else 0
            )
            for count in x
        ]
    )
    script_bars: np.ndarray = np.array(
        [
            (
                map_pages_to_classifications[count]["script"]
                if "script" in map_pages_to_classifications[count]
                else 0
            )
            for count in x
        ]
    )
    other_bars: np.ndarray = np.array(
        [
            (
                map_pages_to_classifications[count]["other"]
                if "other" in map_pages_to_classifications[count]
                else 0
            )
            for count in x
        ]
    )

    proportions_bar_fig: plt.Figure = plt.figure()
    plt.bar(x, synopsis_bars, label="synopsis", color="r", width=1)
    plt.bar(x, script_bars, bottom=synopsis_bars, label="script", color="g", width=1)
    plt.bar(x, other_bars, bottom=synopsis_bars+script_bars, label="other", color="b", width=1)
    plt.legend()
    plt.title("Document Type vs. Document Length")
    plt.xlabel("Document Length (pages)")
    plt.ylabel("Distribution of categories (out of 100%)")
    proportions_bar_fig.show()
    
    input()


if __name__ == "__main__":
    main()
