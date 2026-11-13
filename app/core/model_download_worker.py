"""Download one model in an isolated process so an active transfer can stop."""
import json
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download


ALLOWED_FILES = {'model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json',
                 'vocabulary.txt', 'preprocessor_config.json'}


def download(repo, destination, metadata_path, error_path):
    try:
        info = HfApi().model_info(repo, files_metadata=True)
        sizes = {item.rfilename: item.size for item in info.siblings
                 if item.rfilename in ALLOWED_FILES and item.size is not None}
        Path(metadata_path).write_text(json.dumps({'sizes': sizes}), encoding='utf-8')
        snapshot_download(repo, local_dir=destination, allow_patterns=list(ALLOWED_FILES), max_workers=1)
    except Exception as exc:
        Path(error_path).write_text(str(exc), encoding='utf-8')
        raise


if __name__ == '__main__':
    download(*sys.argv[1:])
