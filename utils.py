import csv
import re
from pathlib import Path
from pprint import pprint
from tqdm import tqdm

def match_transcript(fname: str) -> re.Match | None:
    return re.fullmatch(r"(\w\d{4}\w\d{5})_p(\d+)\.txt", fname)


def is_valid_page(s: str) -> bool:
    return match_transcript(s) is not None


def read_csv(path: Path, header: bool = True) -> list[list[str] | None]:
    try:
        with open(path, "r") as f:
            reader: csv.DictReader = csv.reader(f)

            if header:
                next(reader)

            return list(reader)
    except:
        return []


def get_transcripts(transcript_dir: Path, files: list[str]) -> dict[str, list[str]]:
    transcripts: dict[str, list[str]] = dict()

    print("coalescing transcripts...")
    for fname in tqdm(files):
        match: re.Match = match_transcript(fname)

        id: str = match.group(1)
        page: int = int(match.group(2))

        # add entry to the transcript dict if there is none
        if id not in transcripts:
            transcripts[id] = []

        # files are not necessarily in order, so create space for the files and insert each individually
        num_added_pages: int = page - len(transcripts[id])
        if num_added_pages > 0:
            transcripts[id].extend([""] * num_added_pages)

        # add the content to the newly created space
        with open(transcript_dir / fname, "r") as f:
            transcripts[id][page-1] = f.read()

    return transcripts