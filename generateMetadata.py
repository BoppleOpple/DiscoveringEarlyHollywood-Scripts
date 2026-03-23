import os
import argparse
import random
import re
import json

from pathlib import Path
from typing import Any
from tqdm import tqdm
from pprint import pprint
from ollama import Client, generate, create, delete, GenerateResponse
from typing import Literal, cast
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

The user will provide a transcript and a JSON summarizing the content of the previous pages. Update
this JSON with information from the current page, if applicable. Respond with the updated JSON
file, and be careful to maintain proper JSON syntax.
"""

DECISION_PROMPT = """
You are working as part of a team to exctact correct information about films from the files used to
copyright them. The user has already collected information from each page of the document, but some
of it is conflicting. The information collected includes:

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

The user will provide the section upon which two summaries disagree, the relevant pages, and the
proposed results from each section. Using the content of each of these pages, determing the most
correct value for the provided category. Reply in the following format:

{
    "response": <value>
}
"""


EMPTY_SUMMARY: dict = {
    "title": None,
    "reels": None,
    "author": None,
    "director": None,
    "studio": None,
    "series": None,
    "genres": [],
    "actors": []
}

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
    director: str | None
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
        model="dataDecisionModel",
        from_=args.model,
        system=DECISION_PROMPT
    )

    # fetch the relevant transcripts
    transcripts: dict[str, list[str]] = get_transcripts(args)

    # pprint(transcripts.items())

    # create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    summaries: dict[str, dict] = {}

    """
    I'm kind of running out of simple solutions here, but one really in-depth solution could be to
    take every response, verify that it's a JSON, store every JSON, find the pages that changes
    occurred to each field, so say the title field changes on pages 1, 13, and 18, then find the
    pages that they changed on and feed it back into the LLM and say which one of these answers is
    most correct with the content of each page.
    """

    for id, transcript in tqdm(transcripts.items(), desc="Total"):

        previous_summary: dict = EMPTY_SUMMARY

        modifications: dict[str, list[tuple[int, str]]] = {}

        # summarize each page to save on context
        for page in tqdm(range(len(transcript)), desc="Progress on transcript"):

            last_error: str = None
            while True:
                prompt: str = f"TRANSCRIPT:\n{transcript[page]}\n\nPREVIOUS SUMMARY:{json.dumps(previous_summary)}"

                if last_error:
                    prompt += f"\n(Note: avoid the following JSON exception: \"{e}\")"

                page_response: GenerateResponse = generate(
                    model="pageSummarizationModel",
                    prompt=prompt,
                    stream=False,
                    logprobs=False,
                    think=False
                )
                # log_output(page_response)

                try:
                    parsed_summary: dict = json.loads(page_response.response)
                except Exception as e:
                    print(e)
                    print("encountered exception! retrying...")
                    continue

                break

            for key in parsed_summary.keys():
                if key not in previous_summary:
                    raise Exception("new key created, not sure how to handle")
                if parsed_summary[key] != previous_summary[key]:
                    if key not in modifications:
                        modifications[key] = []
                    
                    modifications[key].append((page, parsed_summary[key]))

            previous_summary = parsed_summary
        
        # print("MODIFICATIONS FOUND:")
        # print(modifications)

        summaries[id] = EMPTY_SUMMARY
        for key in modifications:
            # print(f"finding correct {key}...")

            fmt: str = None
            if key in MetadataObject.model_json_schema()["properties"]:
                fmt = str(MetadataObject.model_json_schema()["properties"][key])
            
            # print(f"detected format {fmt}...")

            prompt = f"Exctract the correct `{key}` with type `{fmt}` from the following pages:\n\n"

            for page, value in modifications[key]:

                prompt += f"PAGE {page}:\n{transcript[page]}\nVALUE FOUND: {value}\n\n"

            while True:
                decision_response: GenerateResponse = generate(
                    model="dataDecisionModel",
                    prompt=prompt,
                    stream=False,
                    logprobs=False,
                    think=True
                )
                # log_output(decision_response)

                try:
                    summaries[id][key] = json.loads(decision_response.response)["response"]
                except Exception as e:
                    print(e)
                    print("encountered exception! retrying...")
                    continue

                break

        # print("FINAL SUMMARY:")
        # print(summaries[id])

        with open(args.outdir / f"{id}.json", "w") as f:
            json.dump(summaries[id], f)


    #     if failed:
    #         with open(args.outdir / f"failed_docs.txt", "a") as f:
    #             print(id, file=f)
    
    delete("pageSummarizationModel")
    delete("dataDecisionModel")


if __name__ == "__main__":
    main()
