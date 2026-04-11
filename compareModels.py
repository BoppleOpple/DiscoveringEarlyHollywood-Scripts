import json
import os
import pathlib
import random
import sys
import time
from ollama import Client, pull

import generateMetadata
import classifyDocuments

if __name__ == "__main__":
    args = generateMetadata.parser.parse_args()

    models = [
        {
            "name": "qwen3:latest",
            "args": {
                "think": False
            }
        },
        # {
        #     "name": "qwen3:latest",
        #     "args": {
        #         "think": True
        #     }
        # },
        # {
        #     "name": "qwen3.5:latest",
        #     "args": {
        #         "think": False
        #     }
        # },
        # {
        #     "name": "qwen3.5:latest",
        #     "args": {
        #         "think": True
        #     }
        # },
        {
            "name": "gemma4:latest",
            "args": {
                "think": False
            }
        },
        # {
        #     "name": "gemma4:latest",
        #     "args": {
        #         "think": True
        #     }
        # },
        # {
        #     "name": "gemma4:26b",
        #     "args": {
        #         "think": False
        #     }
        # },
        # {
        #     "name": "gemma4:26b",
        #     "args": {
        #         "think": True
        #     }
        # },
        # {
        #     "name": "mistral-small3.2:latest"
        # },
    ]

    default_args: dict = generateMetadata.generate_arguments

    seed: int = args.seed if args.seed else random.randint(1, 1000000)
    document_count: int = 40

    _ollama_client: Client = Client(host=args.ollama_host)
    # run metadata generation on all selected models
    for model in models:
        pull(model["name"])

        pathsafe_model: str = model["name"].replace(":", "_")

        outdir: str = None
        if "args" in model and model["args"]:
            pathsafe_args: list[str] = []
            for arg, value in model["args"].items():
                pathsafe_args.append(f"{arg}-{value}")

            subdirectory_name: str = "_".join(pathsafe_args)
            outdir = f"{args.outdir}/compare/{pathsafe_model}/{subdirectory_name}"
        else:
            model["args"] = {}
            outdir = f"{args.outdir}/compare/{pathsafe_model}"

        # load stats from previous session
        stats_path: pathlib.Path = pathlib.Path(outdir) / "stats.json"

        stats: dict = None
        if stats_path.exists():
            with open(stats_path, "r") as json_file:
                stats = json.load(json_file)

        generateMetadata.generate_arguments = {**default_args, **model["args"]}

        start_time_ns: int = time.time_ns()
        generateMetadata.main(str(document_count), *(sys.argv[1:]), "-s", str(seed), "-m", model["name"], "-o", outdir)
        end_time_ns: int = time.time_ns()
        duration_ns: int = end_time_ns - start_time_ns

        metadata_count: int = len(
            [
                fname[:-5]
                for fname in os.listdir(outdir)
                if fname.endswith(".json") and fname != "stats.json"
            ]
        )
        if stats:
            stats["total_time_ns"] += duration_ns
            stats["total_time_s"] = stats["total_time_ns"] * 1e-9

            stats["average_time_ns"] = stats["total_time_ns"] / metadata_count
            stats["average_time_s"] = stats["average_time_ns"] * 1e-9,

            stats["num_docs"] = metadata_count
        else:
            stats = {
                "total_time_ns": duration_ns,
                "total_time_s": duration_ns * 1e-9,
                "average_time_ns": duration_ns / metadata_count,
                "average_time_s": duration_ns / metadata_count * 1e-9,
                "num_docs": metadata_count
            }

        with open(stats_path, "w") as json_file:
            json.dump(stats, json_file, indent=2)
