#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

install_cuda=false
for arg in "$@"; do
    case "$arg" in
        --cuda) install_cuda=true ;;
        *) printf 'Nieznana opcja: %s\nUżycie: bash install.sh [--cuda]\n' "$arg" >&2; exit 2 ;;
    esac
done

if [[ ! -x .venv-py312/bin/python ]]; then
    python_cmd="${WHYSPER_PYTHON:-}"
    if [[ -z "$python_cmd" ]] && command -v python3.12 >/dev/null 2>&1; then
        python_cmd=python3.12
    fi
    if [[ -z "$python_cmd" ]]; then
        printf 'Zainstaluj Python 3.12 z modułem venv i ponów instalację.\n' >&2
        exit 1
    fi
    if ! "$python_cmd" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)'; then
        printf 'Wymagany 64-bitowy Python 3.12.\n' >&2
        exit 1
    fi
    "$python_cmd" -m venv .venv-py312
fi

if ! .venv-py312/bin/python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)'; then
    printf 'Środowisko .venv-py312 wymaga 64-bitowego Pythona 3.12.\n' >&2
    exit 1
fi
.venv-py312/bin/python -m pip install -r requirements.txt
if "$install_cuda"; then
    .venv-py312/bin/python -m pip install 'nvidia-cublas-cu12>=12,<13' 'nvidia-cudnn-cu12>=9,<10'
fi
.venv-py312/bin/python -m pip check
.venv-py312/bin/python -c 'import av, ctranslate2, faster_whisper, sounddevice, sherpa_onnx'
printf 'Instalacja zakończona. Uruchom: bash start.sh\n'
