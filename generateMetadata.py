import os
import argparse
import random
import re
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from ollama import Client, chat, ChatResponse

defaultDir = "/mnt/Database Storage/http/capstone"

SYSTEM_PROMPT = """
Your goal is to extract **exact smippets** from documents in various categories, and to store them
in a JSON object. For each field, follow the provided schema exactly. JSON responses should be of
the form ```json
{
    "title": <str | null>,
    "reels": <int | null>,
    "author": <str | null>,
    "dicrctor": <str | null>,
    "studio": <str | null>,
    "serial": <bool>,
    "genres": [
        <"action" | "comedy" | "drama" | "horror" | "science fiction" | "nonfiction" | "documentary">
    ],
    "actors": [ <str> ],
}
```
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
    "--ollama-host",
    required=False,
    default="http://localhost:11434",
    type=str,
)

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


def main():
    args = parser.parse_args()
    
    _ollama_client = Client(host=args.ollama_host)

    # fetch the relevant transcripts
    transcripts: dict[str, list[str]] = get_transcripts(args)

    # pprint(transcripts.items())

    # create output directory
    os.makedirs(args.outdir, exist_ok=True)

    for id, page_text in tqdm(transcripts.items()):
        failed: bool = True
        for i in range(args.retries):
            response: ChatResponse = chat(
                model=args.model,
                messages=[
                        {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": "\n".join(page_text)
                        }
                    ],
                stream=False,
                logprobs=True,
                think=False
            )

            with open(args.outdir / f"{id}.json", "w") as f:
                if not response.message.content:
                    print(f"no response! retrying... ({i+1})")
                    continue

                failed = False
                print(response.message.content)
                print(response.message.content, file=f)
            
            break

        if failed:
            with open(args.outdir / f"failed_docs.txt", "a") as f:
                print(id, file=f)


if __name__ == "__main__":
    main()
