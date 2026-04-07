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

USER_PROMPT = """
As a leading historian, you have been provided with a historical document describing a film, and
you must group it into one of several categories. These documents could be in any format--i.e.
synopsis, script, etc.

Your goal is to catagorize the document into one of the following categories:
- synopsis (any summary or advertisment of the film)
- script (any direct copy of the text/lines/content of the film)

Reply only with the categorization for the document (i.e. "script" or "synopsis").
"""


parser = argparse.ArgumentParser(
    prog="python3 classifyDocuments.py",
    description="Classifies documents from our capstone's dataset",
    epilog="Copyright Liam Hillery, 2025"
)
parser.add_argument(
    "-o",
    "--out-dir",
    default="out/outliers",
    type=Path
)
parser.add_argument(
    "--counts-csv",
    required=False,
    default=f"out/page_counts/counts.csv",
    type=Path,
)
parser.add_argument(
    "--classifications-csv",
    required=False,
    default=f"out/classifications.csv",
    type=Path,
)

def main():
    args = parser.parse_args()

    # create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # import classifications csv if it exists, otherwise exit with an error
    classifications: dict = {}

    if not args.classifications_csv.exists():
        print(f"could not find {args.classifications_csv}! Exiting...")
        exit(1)

    for row in tqdm(read_csv(args.classifications_csv)):
        classifications[row[0]] = row[1]

    # get the total number of documents in each category...
    category_dist = np.unique_counts(list(classifications.values()))

    # ...and plot them in a pie chart
    proportions_fig: plt.Figure = plt.figure()
    plt.pie(category_dist.counts, labels=category_dist.values)
    proportions_fig.show()

    # import counts csv if it exists, otherwise exit with an error
    page_counts: dict = {}
    # additionally, create a mapping from page count to all ids with that number of pages
    map_pages_to_id: dict[int, list[str]] = {}

    if not args.counts_csv.exists():
        print(f"could not find {args.counts_csv}! Exiting...")
        exit(1)

    for row in tqdm(read_csv(args.counts_csv)):
        id: str = row[0]
        count: int = int(row[1])

        page_counts[id] = count

        # create a list for the given count and add the id
        if count not in map_pages_to_id:
            map_pages_to_id[count] = []

        map_pages_to_id[count].append(id)

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

    proportions_bar_fig: plt.Figure = plt.figure()
    plt.bar(x, synopsis_bars, label="synopsis", color="r", width=1)
    plt.bar(x, script_bars, bottom=synopsis_bars, label="script", color="g", width=1)
    plt.legend()
    plt.title("Document Type vs. Document Length")
    plt.xlabel("Document Length (pages)")
    plt.ylabel("Distribution of categories (out of 100%)")
    proportions_bar_fig.show()


    ids: set = set(page_counts.keys()).intersection(classifications.keys())

    synopsis_ids: list = list(filter(lambda id: classifications[id] == "synopsis", tqdm(ids)))
    script_ids: list = list(filter(lambda id: classifications[id] == "script", tqdm(ids)))

    def _page_length(id: str) -> int:
        return page_counts[id]

    synopsis_ids.sort(key=_page_length, reverse=True)
    script_ids.sort(key=_page_length)

    print("Top 50 longest synopses:")
    with open(args.out_dir / "synopsis_outliers.txt", "w") as f:
        for id in synopsis_ids[:50]:
            f.write(id + "\n")
            print(f"{id} - {page_counts[id]}")
    print()

    print("Top 50 shortest scripts:")
    with open(args.out_dir / "script_outliers.txt", "w") as f:
        for id in script_ids[:50]:
            f.write(id + "\n")
            print(f"{id} - {page_counts[id]}")
    print()

    input()


if __name__ == "__main__":
    main()
