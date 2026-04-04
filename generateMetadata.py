import os
import argparse
import random
import re
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from ollama import Client, generate, create, GenerateResponse
from typing import Literal
from pydantic import BaseModel

from utils import get_transcripts, match_transcript, is_valid_page

defaultDir = "/mnt/Database Storage/http/capstone"

SYSTEM_PROMPT = """
Your goal is to extract **exact smippets** from documents in various categories, and to store them
in a JSON object. For each field, follow the provided schema exactly:

- title: The title of the film described in the document (or null)
- reels: The number of reels described in the document (or null)
- author: The author of the film (or null)
- director: The director of the film (or null)
- studio: The studio responsible for the film (or null)
- series: The name of the series the film is a part of (or null)
- genres: The list of genres that apply to the film
- actors: The list of all **actors** (NOT characters) who act in this film

Respond using the following schema:
{
  "title": str | null,
  "reels": int | null,
  "author": str | null,
  "director": str | null,
  "studio": str | null,
  "series": str | null,
  "genres": ["action" | "comedy" | "drama" | "horror" | "nonfiction"],
  "actors": [ str ]
}
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
    "-s",
    "--seed",
    required=False,
    default=None,
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


def main(*argv: list[str]):
    args = parser.parse_args(argv if argv else None)

    random.seed(args.seed)

    _ollama_client = Client(host=args.ollama_host)
    create(
        model="metadataModel",
        from_=args.model,
        system=SYSTEM_PROMPT
    )

    transcript_files: list[str] = select_files(args)

    # fetch the relevant transcripts
    transcripts: dict[str, list[str]] = get_transcripts(
        args.transcript_dir,
        transcript_files
    )

    # pprint(transcripts.items())

    # create output directory
    os.makedirs(args.outdir, exist_ok=True)

    for id, page_text in tqdm(transcripts.items()):
        failed: bool = True
        for i in range(args.retries):
            response: GenerateResponse = generate(
                model="metadataModel",
                prompt="\n".join(page_text),
                stream=False,
                logprobs=False,
                think=False,
                # format=MetadataObject.model_json_schema()
            )

            with open(args.outdir / f"{id}.json", "w") as f:
                if not response.response:
                    print(f"no response! retrying... ({i+1})")
                    continue

                failed = False
                print(response.response)
                print(response.response, file=f)
            
            break

        if failed:
            with open(args.outdir / f"failed_docs.txt", "a") as f:
                print(id, file=f)


if __name__ == "__main__":
    main()
