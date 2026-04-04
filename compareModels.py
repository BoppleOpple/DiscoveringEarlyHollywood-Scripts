import pathlib
import random
import sys
from ollama import Client, pull

import generateMetadata
import classifyDocuments

if __name__ == "__main__":
    args = generateMetadata.parser.parse_args()

    models = [
        "qwen3:latest",
        "qwen3.5:latest",
        "gemma4:latest",
        "mistral-small3.2:latest",
        "gpt-oss:latest"
    ]

    seed: int = args.seed if args.seed else random.randint(1, 1000000)
    document_count: int = 10

    _ollama_client: Client = Client(host=args.ollama_host)
    # run metadata generation on all selected models
    for model in models:
        pull(model)

        pathsafe_model: str = model.replace(":", "_")
        outdir: str = f"{args.outdir}/compare/{pathsafe_model}"

        generateMetadata.main(str(document_count), *(sys.argv[1:]), "-s", str(seed), "-m", model, "-o", outdir)