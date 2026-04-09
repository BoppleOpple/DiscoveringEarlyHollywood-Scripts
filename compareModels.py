import pathlib
import random
import sys
from ollama import Client, pull

import generateMetadata
import classifyDocuments

if __name__ == "__main__":
    args = generateMetadata.parser.parse_args()

    models = [
        # {
        #     "name": "qwen3:latest",
        #     "args": {
        #         "think": False
        #     }
        # },
        {
            "name": "qwen3:latest",
            "args": {
                "think": True
            }
        },
        # {
        #     "name": "qwen3.5:latest",
        #     "args": {
        #         "think": False
        #     }
        # },
        {
            "name": "qwen3.5:latest",
            "args": {
                "think": True
            }
        },
        # {
        #     "name": "gemma4:latest",
        #     "args": {
        #         "think": False
        #     }
        # },
        {
            "name": "gemma4:latest",
            "args": {
                "think": True
            }
        },
        # {
        #     "name": "mistral-small3.2:latest"
        # },
    ]

    default_args: dict = generateMetadata.generate_arguments


    seed: int = args.seed if args.seed else random.randint(1, 1000000)
    document_count: int = 25

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

        generateMetadata.generate_arguments = {**default_args, **model["args"]}

        generateMetadata.main(str(document_count), *(sys.argv[1:]), "-s", str(seed), "-m", model["name"], "-o", outdir)