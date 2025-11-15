# LoC data parsing scripts
This is a repository of scripts for visualizing and cleaning some data from the library of congress
## Requirements
Since this repo is comprised of python files, python and a few modules stored in `requirements.txt` are the only requirements. The modules can be installed by running
```sh
pip install -r requirements.txt
```
## Usage
### `matchFiles.py`
Detects discrepancies between LoC documents at each stage of processing
```sh
python3 matchFiles.py [-h] [-d DOCUMENT_DIR] [-t TRANSCRIPT_DIR] [-m METADATA_DIR] [-o OUTDIR]
```

### `sampleFiles.py`
Selects files randomly from our capstone's dataset
```sh
python3 sampleFiles.py [-h] [-d DOCUMENT_DIR] [-m METADATA_DIR] [-o OUTFILE] [count]
```