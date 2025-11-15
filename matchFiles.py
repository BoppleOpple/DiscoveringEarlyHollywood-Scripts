import os
import argparse
import zipfile
import random
import re
import json
from PyPDF2 import PdfReader
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
        "-t",
        "--transcript-dir",
        default=f"{defaultDir}/Hollywood_Copright_Materials_Base_Transcriptions"
    )
    parser.add_argument(
        "-m",
        "--metadata-dir",
        default=f"{defaultDir}/cleaned_copyright_with_metadata"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="./out"
    )

    args = parser.parse_args()

    metadata = set(fname[:-24] for fname in os.listdir(args.metadata_dir))

    transcript_dir_textfiles = filter(lambda fname: fname[fname.rindex(".")+1:]=="txt", os.listdir(args.transcript_dir))

    transcripts = set(fname[:fname.rindex(".")] for fname in transcript_dir_textfiles)

    documents = set(os.listdir(args.document_dir))

    counts = {
        "total_missing": None,
        "failed_transcripts": None,
        "failed_metadata": None,
        "missing_pages": None,
    }

    difference = documents.difference(metadata)

    if (not os.path.exists(args.outdir)):
        os.makedirs(args.outdir)

    print("files in documents not in metadata")
    print(list(difference)[:10])
    print(len(difference))

    counts["total_missing"] = len(difference)

    # difference = metadata.difference(documents)

    # print("files in metadata not in documents")
    # print(list(difference)[:10])
    # print(len(difference))

    difference = documents.difference(transcripts)
    
    print("files in documents not in transcripts")
    print(list(difference)[:10])
    print(len(difference))

    counts["failed_transcripts"] = len(difference)

    with zipfile.ZipFile(f"{args.outdir}/failed_transcripts.zip", "w") as zf:
        for doc_id in tqdm(difference):
            document_path = f"{args.document_dir}/{doc_id}/{doc_id}.pdf"
            if (os.path.exists(document_path)):
                zf.write(document_path, f"{doc_id}/document.pdf")

    # difference = transcripts.difference(documents)

    # print("files in transcripts not in documents")
    # print(list(difference)[:10])
    # print(len(difference))

    difference = transcripts.difference(metadata)
    
    print("files in transcripts not in metadata")
    print(list(difference)[:10])
    print(len(difference))

    counts["failed_metadata"] = len(difference)

    with zipfile.ZipFile(f"{args.outdir}/failed_metadata.zip", "w") as zf:
        for doc_id in tqdm(difference):
            document_path = f"{args.document_dir}/{doc_id}/{doc_id}.pdf"
            transcript_path = f"{args.transcript_dir}/{doc_id}.txt"

            if (os.path.exists(document_path)):
                zf.write(document_path, f"{doc_id}/document.pdf")
            
            if (os.path.exists(transcript_path)):
                zf.write(transcript_path, f"{doc_id}/transcript.txt")

    # difference = metadata.difference(transcripts)

    # print("files in metadata not in transcripts")
    # print(list(difference)[:10])
    # print(len(difference))

    missing_pages = []

    with zipfile.ZipFile(f"{args.outdir}/missing_pages.zip", "w") as zf:
        for doc_id in tqdm(metadata):
            document_path = f"{args.document_dir}/{doc_id}/{doc_id}.pdf"
            transcript_path = f"{args.transcript_dir}/{doc_id}.txt"
            metadata_path = f"{args.metadata_dir}/{doc_id}with_added_metadata.json"

            if (not os.path.exists(document_path)):
                document_path = f"{args.document_dir}/{doc_id}/{doc_id}.PDF"

            reader = PdfReader(document_path)
            number_of_pages = len(reader.pages)

            transcribed_pages = 0
            with open(metadata_path, "r") as metadata:
                transcribed_pages = len(json.load(metadata)["text"])

            discrepancy = number_of_pages - transcribed_pages
            if (discrepancy > 0):
                missing_pages.append(doc_id)

                document_dir = f"missing_{discrepancy}_{"pages" if discrepancy > 1 else "page"}/{doc_id}"
                if (os.path.exists(document_path)):
                    zf.write(document_path, f"{document_dir}/document.pdf")
                
                if (os.path.exists(transcript_path)):
                    zf.write(transcript_path, f"{document_dir}/transcript.txt")
                
                if (os.path.exists(metadata_path)):
                    zf.write(metadata_path, f"{document_dir}/metadata.json")
    
    print("cleaned transcripts missing pages")
    print(missing_pages[:10])
    print(len(missing_pages))

    counts["missing_pages"] = len(missing_pages)

    with open(f"{args.outdir}/tallies.json", "r") as tallies_json:
        json.dump(counts, tallies_json)



if __name__ == "__main__":
    main()
