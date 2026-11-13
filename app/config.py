from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
UPLOADS = DATA / 'uploads'
RECORDINGS = DATA / 'recordings'
LOGS = ROOT / 'logs'
DB_PATH = DATA / 'whysper.sqlite3'

for directory in (DATA, UPLOADS, RECORDINGS, LOGS, ROOT / 'models', ROOT / 'output'):
    directory.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    model: Literal['tiny', 'base', 'small', 'medium', 'large-v3', 'turbo'] = 'large-v3'
    language: str = ''
    device: Literal['cuda', 'cpu'] = 'cuda'
    gpu_id: str = Field(default='', max_length=128)
    compute_type: Literal['auto', 'float16', 'int8_float16'] = 'auto'
    task: Literal['transcribe', 'translate'] = 'transcribe'
    vad: bool = True
    diarization: bool = False
    beam_size: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.0, ge=0, le=1)
    best_of: int = Field(default=5, ge=1, le=10)
    patience: float = Field(default=1.0, ge=0.1, le=5.0)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=3.0)
    no_repeat_ngram_size: int = Field(default=0, ge=0, le=10)
    no_speech_threshold: float = Field(default=0.6, ge=0, le=1)
    condition_on_previous_text: bool = True
    vad_threshold: float = Field(default=0.5, ge=0.05, le=0.95)
    vad_min_silence_ms: int = Field(default=2000, ge=100, le=10000)
    vad_speech_pad_ms: int = Field(default=400, ge=0, le=1500)
    word_timestamps: bool = False
    initial_prompt: str = Field(default='', max_length=4000)
    hotwords: str = Field(default='', max_length=4000)
    cache_dir: str = str(ROOT / 'models')
    output_dir: str = str(ROOT / 'output')
    theme: Literal['dark', 'light'] = 'dark'
    workers: int = Field(default=1, ge=1, le=2)
    keep_model: bool = False
    max_upload_gb: int = Field(default=20, ge=1, le=100)


MODEL_REPOS = {
    'tiny': 'Systran/faster-whisper-tiny',
    'base': 'Systran/faster-whisper-base',
    'small': 'Systran/faster-whisper-small',
    'medium': 'Systran/faster-whisper-medium',
    'large-v3': 'Systran/faster-whisper-large-v3',
    'turbo': 'deepdml/faster-whisper-large-v3-turbo-ct2',
}

MODEL_SIZES = {
    'tiny': 'ok. 75 MB', 'base': 'ok. 145 MB', 'small': 'ok. 0,5 GB',
    'medium': 'ok. 1,5 GB', 'large-v3': 'ok. 3,1 GB', 'turbo': 'ok. 1,6 GB',
}
