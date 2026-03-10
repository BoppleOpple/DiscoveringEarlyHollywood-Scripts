"""A CLI program that uploads LoC data from the local filesystem to a PostgreSQL database."""

import os
import argparse
import json
import re
import zipfile
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(
    prog="collectValidFiles.py",
    description="A program that collects all successfully transcribed documents for DEH",
    epilog="Willy N' Gang, 2025",
)

parser.add_argument(
    "-d", "--document-dir", default="./data/film_copyright", type=Path
)
parser.add_argument(
    "-t",
    "--transcript-dir",
    default="./data/Hollywood_Copyright_Materials_Base_Transcriptions",
    type=Path,
)
parser.add_argument(
    "-m",
    "--metadata-dir",
    required=False,
    default="./data/cleaned_copyright_with_metadata",
    type=Path,
)
parser.add_argument(
    "-a",
    "--analysis-dir",
    required=False,
    default="./data/no_shot_gpt_analysis",
    type=Path,
)
parser.add_argument(
    "-o",
    "--outdir",
    required=False,
    default="./out",
    type=Path,
)


# TODO improve code commenting post-prototype
def formatLLMAnalysis(analysis: dict) -> dict:
    """Form a ``dict`` from metadata extracted by an LLM.

    Parameters
    ----------
    analysis : dict
        A ``dict`` of the form::

            {
                "File_Name": A string containing the document id,
                "text": A string containing the transcript of the document,
                "response": A string containing:
                    - a JSON of a single analysis
                    - a JSON of an array of analyses
                    - a markdown block for either above JSON
                    - several markdown blocks for either above JSON
            }

    Returns
    -------
    formattedAnalysis : dict
        A ``dict`` of the form::

            {
                "title": str,
                "actors": list[str],
                "failed": bool
            }
    """
    formattedAnalysis: dict = {"title": None, "actors": [], "failed": False}
    if analysis:
        try:
            responses = analysis["response"].split("```json")

            responseList = []

            for response in responses:
                # locate the JSON content
                minCharacter: int = min(
                    response.index("{") if "{" in response else len(response),
                    response.index("[") if "[" in response else len(response),
                )
                maxCharacter: int = max(
                    response.rindex("}") if "}" in response else 0,
                    response.rindex("]") if "]" in response else 0,
                )

                # ignore the string if there are no JSON opening or closing brackets
                if minCharacter == len(response) or maxCharacter == 0:
                    continue

                cleanedResponse, n = re.subn(
                    r",(?=\s*[\}\]])",
                    "",
                    response[minCharacter : maxCharacter + 1],  # noqa E203
                )

                # LLM allows comments in JSON files
                cleanedResponse, n = re.subn(r"//.*\n", "", cleanedResponse)

                # account for multiple documents being stored in the same JSON object
                responseObject = json.loads(cleanedResponse)
                if type(responseObject) is dict:
                    if "Films" not in responseObject:
                        responseList.append(responseObject)
                    else:
                        responseList.extend(responseObject["Films"])
                elif type(responseObject) is list:
                    responseList.extend(responseObject)

            if len(responseList) == 1:
                responseDict = responseList[0]
                # parse "Title" field
                if "Title" in responseDict:
                    formattedAnalysis["title"] = responseDict["Title"]
                else:
                    print(
                        f"Analysis of {analysis["File_Name"]} is missing key 'Title': {responseDict}"  # noqa E501
                    )

                # parse "Actors" field
                if "Actors" in responseDict:
                    actors = responseDict["Actors"]

                    # no actors can be represented as "N/A", ["N/A"], or []
                    if type(actors) is list:
                        if "N/A" not in actors:
                            formattedAnalysis["actors"] = actors
                        else:
                            formattedAnalysis["actors"] = []

                    elif type(actors) is str:
                        if actors != "N/A":
                            # Assume the string contains the actor's name
                            formattedAnalysis["actors"] = [actors]
                        else:
                            formattedAnalysis["actors"] = []
            else:
                formattedAnalysis["failed"] = True
                print(
                    f"Analysis of {analysis["File_Name"]} includes {len(responseList)} documents (expected 1)"  # noqa E501
                )

        except json.decoder.JSONDecodeError as e:
            formattedAnalysis["failed"] = True
            print(f"Error while decoding {analysis["File_Name"]}:")
            print(e)
    else:
        formattedAnalysis["failed"] = True

    return formattedAnalysis if not formattedAnalysis["failed"] else None

def idIsValid(id: str) -> bool:
    return re.fullmatch(r"\w\d{4}\w\d{5}", id) is not None


def main(argv=None):
    """Collect data from specified locations"""
    args = parser.parse_args(argv)

    print("parsing movies")
    ids: list[str] = [fname[:-5] for fname in os.listdir(args.analysis_dir)]

    ids = list(filter(idIsValid, ids))

    os.makedirs(args.outdir, exist_ok=True)

    with zipfile.ZipFile(args.outdir / "valid_documents.zip", "w") as zf:
        # iterate through every document id that has been analyzed
        for document_id in tqdm(ids):
            documentFile: Path = args.document_dir / f"{document_id}/{document_id}.pdf"
            transcriptFile: Path = args.transcript_dir / f"{document_id}.txt"
            metadataFile: Path = args.metadata_dir / f"{document_id}with_added_metadata.json"
            analysisFile: Path = args.analysis_dir / f"{document_id}.json"

            invalid = False
            for file in (documentFile, transcriptFile, metadataFile, analysisFile):
                if not file.exists():
                    invalid = True

            if invalid: continue

            analysis: dict = None
            with open(analysisFile, "r") as analysisJson:
                analysis = json.load(analysisJson)

            formattedAnalysis: dict = formatLLMAnalysis(analysis)

            if formattedAnalysis:
                zipDocumentDir = f"film_copyright/{document_id}/{documentFile.name}"
                zipTranscriptDir = f"Hollywood_Copyright_Materials_Base_Transcriptions/{transcriptFile.name}"
                zipMetadataDir = f"cleaned_copyright_with_metadata/{metadataFile.name}"
                zipAnalysisDir = f"no_shot_gpt_analysis/{analysisFile.name}"

                if (os.path.exists(documentFile)):
                    zf.write(documentFile, zipDocumentDir)

                if (os.path.exists(transcriptFile)):
                    zf.write(transcriptFile, zipTranscriptDir)

                if (os.path.exists(metadataFile)):
                    zf.write(metadataFile, zipMetadataDir)

                if (os.path.exists(analysisFile)):
                    zf.write(analysisFile, zipAnalysisDir)



if __name__ == "__main__":
    main()
