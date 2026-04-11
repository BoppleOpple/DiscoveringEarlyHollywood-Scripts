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
As a leading historian, you have been provided with a historical document describing a film, and
you must extract **exact snippets** from documents in various categories. These snippets should be presented
in a JSON object with the following schema:

{
  "title": str | null,
  "producer": str | null,
  "writer": str | null,
  "production_company": str | null,
  "reels": int | null,
  "is_serial": bool,
  "series": str | null,
  "genres": ["action" | "comedy" | "drama" | "horror" | "nonfiction"],
  "characters": [
    {
      "character_description": str | null,
      "character_name": str | null,
      "actor": str | null
    }
  ],
  "locations": [
    {
      "location_name": str | null,
      "location_description": str | null
    }
  ]
}

Here is a description of each field:

- title: The title of the film described in the document (or null)
- producer: The producer of the film (or null)
- writer: The writer of the film (or null)
- production_company: The production company responsible for the film (or null)
- reels: The number of reels described in the document (or null)
- is_serial: true if the film is part of a series, false otherwise
- series: The name of the series the film is a part of (or null if `isSerial` is false)
- genres: The list of genres that apply to the film
- characters: The list of all characters present in this film. Each element should have the
  following fields:
  - character_description: A brief description of this character, including their name (i.e.
    "<character name> does [...]") (or null)
  - character_name: The name of the character in the described film (or null)
  - actor: The name of actor or actress who plays this character (or null)
- The list of all locations present in this film. Each element should have the
  following fields:
  - location_name: The name of the location/setting (or null if no name can be found)
  - location_description: A brief description of this location (or null)

Respond with ONLY the JSON object.
"""

generate_arguments = {
    "think": False,
    # "format": MetadataObject.model_json_schema()
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
parser.add_argument(
    "-i",
    "--id-file",
    required=False,
    default=None,
    type=Path,
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


# select files based on CLI filters
def select_files(args: argparse.Namespace) -> list[str]:
    files: list[str] = os.listdir(args.transcript_dir)

    print(f"found {len(files)} files and directories")
    # print(files[:10])
    
    valid_pages: list[str] = list(filter(is_valid_page, files))

    print(f"filtered to {len(valid_pages)} transcript files")
    # print(valid_pages[:10])

    ids: tuple[str] = None

    if args.id_file and args.id_file.exists():
        with open(args.id_file, "r") as f:
            ids = tuple(line.strip() for line in f.readlines() if line.strip() != "")
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
        if (args.outdir / f"{id}.json").exists():
            continue

        failed: bool = True
        for i in range(args.retries):
            response: GenerateResponse = generate(
                model="metadataModel",
                prompt="\n".join(page_text),
                stream=False,
                logprobs=False,
                **generate_arguments
            )

            with open(args.outdir / f"{id}.json", "w") as f:
                if not response.response:
                    print(f"no response! retrying... ({i+1})")
                    continue
                
                # remove markdown if present
                response_lines: list[str] = response.response.splitlines()

                if response_lines[0].startswith("```"):
                    response_lines.pop(0)
                
                if response_lines[-1].startswith("```"):
                    response_lines.pop(-1)

                response_text: str = "\n".join(response_lines)

                failed = False
                # print(response_text)
                f.write(response_text)

            break

        if failed:
            with open(args.outdir / f"failed_docs.txt", "a") as f:
                print(id, file=f)


if __name__ == "__main__":
    main()
