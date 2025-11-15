import os
import argparse
import zipfile
import random

defaultDir = "/mnt/Database Storage/http/capstone"

def main():
    parser = argparse.ArgumentParser(
        prog="python3 sampleFiles.py",
        description="Selects files randomly from our capstone's dataset",
        epilog="Copyright Liam Hillery, 2025"
    )

    parser.add_argument("count", nargs="?", default=20)
    parser.add_argument(
        "-d",
        "--document-dir",
        default=f"{defaultDir}/film_copyright"
    )
    parser.add_argument(
        "-m",
        "--metadata-dir",
        default=f"{defaultDir}/cleaned_copyright_with_metadata"
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="sample.zip"
    )

    args = parser.parse_args()

    metadata = random.sample(os.listdir(args.metadata_dir), args.count)
    documents = [fname[:-24] for fname in metadata]

    with zipfile.ZipFile(args.outfile, "w") as myzip:
        for f in documents:
            document_dir = f"{args.document_dir}/{f}"
            for document in os.listdir(document_dir):
                myzip.write(f"{document_dir}/{document}")
            myzip.write(f"{args.metadata_dir}/{f}with_added_metadata.json")


if __name__ == "__main__":
    main()
