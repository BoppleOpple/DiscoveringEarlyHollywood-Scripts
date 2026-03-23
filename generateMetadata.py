import os
import argparse
import random
import re
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from ollama import Client, generate, create, delete, GenerateResponse
from typing import Literal
from pydantic import BaseModel

defaultDir = "/mnt/Database Storage/http/capstone"

PAGE_PROMPT = """
Your goal is to extract **exact snippets** the provided page of a document, and to store them
in a JSON object. If you are unsure of the accuracy of your results, err on the side of returning
"null" than returning inaccurate information. You will summarize this page along the following
criteria and ONLY the following criteria:

- title: The title of the film described in the document. If unclear or unstated, this section
should be null.
- reels: The number of reels described in the document. If unclear or unstated, this section should
be null.
- author: The author of the film. If unclear or unstated, this section should be null.
- director: The director of the film. If unclear or unstated, this section should be null.
- studio: The studio responsible for the film. If unclear or unstated, this section should be
null.
- series: The name of the series the film is a part of. If unclear or unstated, this section should
be null.
- genres: The list of genres that apply to the film. If unclear or unstated, this section should be
null.
- actors: The list of all actors who act in this film. There is a difference between an actor ana a
character. This list should ONLY include the names of real people who play the role of their
respective characters in this film. DO NOT UNDER ANY CIRCUMSTANCES name a character instead of an
actor. If unclear or unstated, this section should be null.
"""

COALTION_PROMPT = """
You are processing historical film data for a team to review at a later date. I have read through
the transcript of a film or film review page-by-page, and have summarized each on the following
criteria:

- title: The title of the film described in the document. If unclear or unstated, this section
should be null.
- reels: The number of reels described in the document. If unclear or unstated, this section should
be null.
- author: The author of the film. If unclear or unstated, this section should be null.
- director: The director of the film. If unclear or unstated, this section should be null.
- studio: The studio responsible for the film. If unclear or unstated, this section should be
null.
- series: The name of the series the film is a part of. If unclear or unstated, this section should
be null.
- genres: The list of genres that apply to the film. If unclear or unstated, this section should be
null.
- actors: The list of all unique actors who act in this film. If unclear or unstated,
this section should be null.

Your job is to coalesce these summaries into a JSON file contianing information about the film as a
whole. The worst case scenario is including information that is incorrect, so follow the below
schema with no additional discussion and return "null" (or an empty list where applicable) if any
information is unclear:

{
    "title": String | null
    "reels": int | null
    "author": String | null
    "dicrctor": String | null
    "studio": String | null
    "series": String | null
    "genres": UniqueArray[
        "action" |
        "comedy" |
        "drama" |
        "horror" |
        "science fiction" |
        "nonfiction" |
        "documentary"
    ]
    "actors": UniqueArray[String]
}

This task does not involve including all information from each page individually, but rather
drawing information from multiple summaries at once.
"""


parser = argparse.ArgumentParser(
    prog="python3 generateMetadata.py",
    description="Generates [count] metadata files for our capstone's dataset",
    epilog="Copyright Liam Hillery, 2025"
)

parser.add_argument("count", nargs="?", default=20, type=int)
parser.add_argument(
    "-o",
    "--outdir",
    default="out/generated_metadata",
    type=Path
)
parser.add_argument(
    "-t",
    "--transcript-dir",
    required=False,
    default=f"{defaultDir}/qwen_ocr",
    type=Path,
)

# largest document is s1229l11579 (216 pages)
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

class MetadataObject (BaseModel):
    title: str | None
    reels: int | None
    author: str | None
    dicrctor: str | None
    studio: str | None
    series: str | None
    genres: list[Literal["action", "comedy", "drama", "horror", "science fiction", "nonfiction", "documentary"]]
    actors: list[str]


def match_transcript(fname: str) -> re.Match | None:
    return re.fullmatch(r"(\w\d{4}\w\d{5})_p(\d+)\.txt", fname)

def is_valid_page(s: str) -> bool:
    return match_transcript(s) is not None


# select files based on CLI filters
def select_files(args: argparse.Namespace) -> list[str]:
    files: list[str] = os.listdir(args.transcript_dir)

    print(f"found {len(files)} files and directories")
    # print(files[:10])
    
    valid_pages: list[str] = list(filter(is_valid_page, files))

    print(f"filtered to {len(valid_pages)} transcript files")
    # print(valid_pages[:10])

    ids: tuple[str] = None

    if args.id:
        ids = (args.id,)
    else:
        all_ids = [match_transcript(s).group(1) for s in valid_pages]

        if args.count > 0:
            ids = tuple(random.sample(all_ids, args.count))
        else:
            ids = tuple(all_ids)

    sample: list[str] = list(filter(lambda fname: fname.startswith(ids), valid_pages))

    print(f"final sample ({len(sample)} files):")
    print(sample)

    return sample


def get_transcripts(args: argparse.Namespace) -> dict[str, list[str]]:
    # select the files to operate on
    files: list[str] = select_files(args)

    transcripts: dict[str, list[str]] = dict()

    print("coalescing transcripts...")
    for fname in tqdm(files):
        match: re.Match = match_transcript(fname)

        id: str = match.group(1)
        page: int = int(match.group(2))

        # add entry to the transcript dict if there is none
        if id not in transcripts:
            transcripts[id] = []

        num_added_pages: int = page - len(transcripts[id])
        if num_added_pages > 0:
            transcripts[id].extend([None] * num_added_pages)

        # add the content to the newly created space
        with open(args.transcript_dir / fname, "r") as f:
            transcripts[id][page-1] = f.read()

    return transcripts

def log_output(response: GenerateResponse):
    print("RESPONSE:")
    print(response.response)

    print()
    print(f"USED CONTEXT: {len(response.context)}")


def main():
    args = parser.parse_args()
    
    _ollama_client = Client(host=args.ollama_host)

    create(
        model="pageSummarizationModel",
        from_=args.model,
        system=PAGE_PROMPT
    )

    create(
        model="pageCoalitionModel",
        from_=args.model,
        system=COALTION_PROMPT
    )

    # fetch the relevant transcripts
    transcripts: dict[str, list[str]] = get_transcripts(args)

    # pprint(transcripts.items())

    # create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    summaries: dict[str, list[str]] = {}

    """
    I'm kind of running out of simple solutions here, but one really in-depth solution could be to
    take every response, verify that it's a JSON, store every JSON, find the pages that changes
    occurred to each field, so say the title field changes on pages 1, 13, and 18, then find the
    pages that they changed on and feed it back into the LLM and say which one of these answers is
    most correct with the content of each page.
    """

    for id, transcript in tqdm(transcripts.items(), desc="Summaries"):
        summaries[id] = []

        # summarize each page to save on context
        for page in tqdm(transcript, desc="Progress on transcript"):

            page_response: GenerateResponse = generate(
                model="pageSummarizationModel",
                prompt=page,
                stream=False,
                logprobs=False,
                think=False
            )
            log_output(page_response)

            summaries[id].append(page_response.response)

    for id, transcript in tqdm(transcripts.items(), desc="Postprocessing"):
        failed: bool = True
        for i in range(args.retries):
            
            coalition_response: GenerateResponse = generate(
                model="pageCoalitionModel",
                prompt="\n---NEXT PAGE---\n".join(summaries[id]),
                stream=False,
                logprobs=False,
                think=False,
                format=MetadataObject.model_json_schema()
            )


            with open(args.outdir / f"{id}.json", "w") as f:
                if not coalition_response.response:
                    print(f"no response! retrying... ({i+1})")
                    continue

                failed = False

                log_output(coalition_response)
                print(coalition_response.response, file=f)

            break

        if failed:
            with open(args.outdir / f"failed_docs.txt", "a") as f:
                print(id, file=f)
    
    delete("pageSummarizationModel")
    delete("pageCoalitionModel")


if __name__ == "__main__":
    main()
