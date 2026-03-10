import os
import argparse
import zipfile
import random
from pathlib import Path
from tqdm import tqdm

defaultDir = "/mnt/Database Storage/http/capstone"

def main():
    parser = argparse.ArgumentParser(
        prog="python3 sampleFiles.py",
        description="Selects files randomly from our capstone's dataset",
        epilog="Copyright Liam Hillery, 2025"
    )

    parser.add_argument("count", nargs="?", default=20, type=int)
    parser.add_argument(
        "-o",
        "--outfile",
        default="sample.zip",
    )
    parser.add_argument(
        "-d",
        "--document-dir",
        default=f"{defaultDir}/film_copyright",
        type=Path,
    )
    parser.add_argument(
        "-t",
        "--transcript-dir",
        required=False,
        default="./data/Hollywood_Copyright_Materials_Base_Transcriptions",
        type=Path,
    )
    parser.add_argument(
        "-m",
        "--metadata-dir",
        required=False,
        default=f"{defaultDir}/cleaned_copyright_with_metadata",
        type=Path,
    )
    parser.add_argument(
        "-a",
        "--analysis-dir",
        required=False,
        default="./data/no_shot_gpt_analysis",
        type=Path,
    )

    args = parser.parse_args()

    ids = random.sample(os.listdir(args.document_dir), args.count)
    with zipfile.ZipFile(args.outfile, "w") as myzip:
        for id in tqdm(ids):
            document_dir = args.document_dir / id
            for document in os.listdir(document_dir):
                myzip.write(f"{document_dir}/{document}")

            if args.transcript_dir.exists():
                myzip.write(args.transcript_dir / f"{id}.txt")

            if args.metadata_dir.exists():
                myzip.write(args.metadata_dir / f"{id}with_added_metadata.json")

            if args.analysis_dir.exists():
                myzip.write(args.analysis_dir / f"{id}.json")


if __name__ == "__main__":
    main()
