import os
import argparse
import zipfile
import random
import re
import json
import pathlib
import pdf2image.pdf2image
import matplotlib.pyplot as mpl
import numpy as np
from tqdm import tqdm

defaultDir = "/mnt/Database Storage/http/capstone"

def main():
    parser = argparse.ArgumentParser(
        prog="python3 matchFiles.py",
        description="Detects discrepancies between LoC documents at each stage of processing",
        epilog="Copyright Liam Hillery, 2025"
    )

    parser.add_argument(
        "-d",
        "--document-dir",
        default=f"{defaultDir}/film_copyright"
    )

    parser.add_argument(
        "-o",
        "--outfile",
        default=f"out/counts.csv"
    )

    args = parser.parse_args()

    documents = set(os.listdir(args.document_dir))


    counts = {}
    max_id = ""
    max_pages = 0

    for doc_id in tqdm(documents):
        try:
            doc_dir = pathlib.Path(args.document_dir) / doc_id

            document = doc_dir / f"{doc_id}.pdf"

            if not document.exists():
                document = doc_dir / f"{doc_id}.PDF"

            info = pdf2image.pdfinfo_from_path(document)
            counts[doc_id] = info["Pages"]

            if max_pages < info["Pages"]:
                max_id = doc_id
                max_pages = info["Pages"]
        except Exception as e:
            print(e)
    
    page_counts, document_counts = np.unique_counts(list(counts.values()))

    print("total pages")
    print(sum(counts.values()))

    print("max")
    print(max_id, max_pages)

    mpl.hist(counts.values())

    mpl.show()

    with open(args.outfile, "w") as f:
        f.write("num_documents,page_count\n")
        for i in range(len(page_counts)):
            f.write(f"{document_counts[i]},{page_counts[i]}\n")


if __name__ == "__main__":
    main()