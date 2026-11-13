#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv-py312/bin/python ]]; then
    bash ./install.sh
fi
if ! .venv-py312/bin/python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)'; then
    printf 'Środowisko .venv-py312 wymaga 64-bitowego Pythona 3.12.\n' >&2
    exit 1
fi
# The ELF loader reads LD_LIBRARY_PATH at process startup. Include libraries
# installed by NVIDIA's pip wheels before launching Python/CTranslate2.
cuda_paths="$(.venv-py312/bin/python -c 'import os, site; print(os.pathsep.join(os.path.join(p, "nvidia", n, "lib") for p in site.getsitepackages() for n in ("cublas", "cudnn") if os.path.isdir(os.path.join(p, "nvidia", n, "lib"))))')"
if [[ -n "$cuda_paths" ]]; then
    export LD_LIBRARY_PATH="$cuda_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec .venv-py312/bin/python -m app.main
