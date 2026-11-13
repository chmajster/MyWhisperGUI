"""whisper.cpp Vulkan adapter. Models and inference stay local to the machine."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import wave

import numpy as np

from app.config import ROOT, VULKAN_MODELS
from app.core import audio_processor
from app.core.speaker_diarization import SpeakerCancelled, label_segments
from app.core.whisper_engine import Cancelled


def whisper_cli():
    configured = os.environ.get('WHYSPER_WHISPER_CLI')
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path.resolve())
        raise RuntimeError('WHYSPER_WHISPER_CLI nie wskazuje istniejącego programu whisper-cli.')
    executable = 'whisper-cli.exe' if os.name == 'nt' else 'whisper-cli'
    for path in (ROOT / 'tools' / 'whisper.cpp' / 'build' / 'bin' / executable,
                 ROOT / 'tools' / 'whisper.cpp' / 'build' / 'bin' / 'Release' / executable):
        if path.is_file():
            return str(path.resolve())
    found = shutil.which('whisper-cli')
    if found:
        return found
    raise RuntimeError('Nie znaleziono whisper-cli z obsługą Vulkan. Zbuduj whisper.cpp zgodnie z README lub ustaw WHYSPER_WHISPER_CLI.')


def _speech_ranges(samples, options):
    if not options.vad:
        return [(0, len(samples))]
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    vad = VadOptions(threshold=options.vad_threshold, min_silence_duration_ms=options.vad_min_silence_ms,
                     speech_pad_ms=options.vad_speech_pad_ms)
    return [(max(0, part['start']), min(len(samples), part['end']))
            for part in get_speech_timestamps(samples, vad) if part['end'] > part['start']]


class VulkanEngine:
    def __init__(self, models):
        self.models = models
        self.guard = threading.RLock()
        self.loaded_name = None
        self.loaded_gpu = None

    def unload(self):
        # whisper-cli owns its model only for the lifetime of each subprocess.
        self.loaded_name = self.loaded_gpu = None

    @staticmethod
    def check_prerequisites():
        return whisper_cli()

    def prepare(self, options):
        if options.task == 'translate' and options.model == 'turbo':
            raise ValueError('Model turbo nie jest przeznaczony do tłumaczenia. Wybierz np. large-v3.')
        if (options.compute_type != 'auto' or options.patience != 1.0 or options.repetition_penalty != 1.0
                or options.no_repeat_ngram_size != 0 or options.hotwords or not options.condition_on_previous_text):
            raise ValueError('Wybrane parametry są dostępne tylko w faster-whisper (CPU/CUDA), nie w Vulkan.')
        self.check_prerequisites()
        path = self.models.path(options.model, options.cache_dir, 'vulkan') / VULKAN_MODELS[options.model]
        if not path.is_file():
            raise RuntimeError('Brak modelu GGML dla Vulkan. Pobierz go w zakładce Modele AI.')
        return path

    def transcribe(self, path, options, on_progress, cancelled, on_segment=None, on_started=None, speaker_tracker=None):
        return self._transcribe(audio_processor.chunks(path), options, on_progress, cancelled,
                                on_segment, on_started, speaker_tracker)

    def transcribe_samples(self, samples, options, cancelled):
        return self._transcribe([(0.0, samples)], options, lambda _: None, cancelled)

    def _transcribe(self, chunks, options, on_progress, cancelled, on_segment=None, on_started=None, speaker_tracker=None):
        with self.guard:
            model_path = self.prepare(options)
            result = {'segments': [], 'language': None, 'device': 'vulkan', 'model': options.model,
                      'compute_type': 'ggml', 'gpu_index': options.vulkan_device}
            if speaker_tracker:
                result['diarization'] = True
            language = options.language or None
            has_audio = False
            if on_started:
                on_started()
            try:
                for offset, samples in chunks:
                    has_audio = True
                    if cancelled():
                        raise Cancelled()
                    pending = []
                    for start, end in _speech_ranges(samples, options):
                        if cancelled():
                            raise Cancelled()
                        data = self._run_chunk(samples[start:end], model_path, options, language, cancelled,
                                               bool(options.word_timestamps or speaker_tracker))
                        language = language or data.get('result', {}).get('language')
                        result['language'] = language
                        for item in data.get('transcription', []):
                            text = item.get('text', '').strip()
                            if not text:
                                continue
                            times = item['offsets']
                            base = offset + start / 16000
                            segment = {'start': base + times['from'] / 1000,
                                       'end': base + times['to'] / 1000, 'text': text}
                            if options.word_timestamps or speaker_tracker:
                                segment['words'] = [{'start': segment['start'], 'end': segment['end'],
                                                     'word': text, 'probability': None}]
                            pending.append(segment)
                    if speaker_tracker and pending:
                        turns = speaker_tracker.turns(samples, cancelled)
                        pending = label_segments(pending, turns, offset, keep_words=options.word_timestamps)
                    for item in pending:
                        result['segments'].append(item)
                        if on_segment:
                            on_segment(item)
                        on_progress(item['end'])
                    on_progress(offset + len(samples) / 16000)
                if not has_audio:
                    raise ValueError('Nagranie nie zawiera żadnych próbek audio do przetworzenia.')
                if cancelled():
                    raise Cancelled()
                return result
            except SpeakerCancelled:
                raise Cancelled() from None

    def _run_chunk(self, samples, model_path, options, language, cancelled, words):
        with tempfile.TemporaryDirectory(prefix='whysper-vulkan-') as directory:
            root = Path(directory)
            audio, output, log_path = root / 'audio.wav', root / 'result', root / 'cli.log'
            pcm = (np.clip(samples, -1, 1) * 32767).astype('<i2')
            with wave.open(str(audio), 'wb') as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(16000)
                target.writeframes(pcm.tobytes())
            command = [whisper_cli(), '--model', str(model_path), '--file', str(audio), '--output-json',
                       '--output-file', str(output), '--language', language or 'auto', '--device', str(options.vulkan_device),
                       '--beam-size', str(options.beam_size), '--best-of', str(options.best_of),
                       '--temperature', str(options.temperature), '--no-fallback',
                       '--no-speech-thold', str(options.no_speech_threshold)]
            if options.task == 'translate':
                command.append('--translate')
            if options.initial_prompt:
                command.extend(['--prompt', options.initial_prompt])
            if words:
                command.extend(['--max-len', '1', '--split-on-word'])
            with log_path.open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, stdout=log, stderr=log,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                try:
                    while process.poll() is None:
                        if cancelled():
                            raise Cancelled()
                        try:
                            process.wait(timeout=.2)
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
            log = log_path.read_text(encoding='utf-8', errors='replace')
            if process.returncode:
                raise RuntimeError(f'whisper-cli zakończył pracę z błędem ({process.returncode}): {log[-3000:]}')
            initialized = re.search(r'whisper_backend_init_gpu:\s+using\s+Vulkan(\d*)\s+backend', log, re.IGNORECASE)
            failed = re.search(r'failed to initialize\s+Vulkan\d*\s+backend', log, re.IGNORECASE)
            if not initialized or failed:
                raise RuntimeError('whisper-cli nie potwierdził uruchomienia na Vulkan. Sprawdź wersję programu i sterownik GPU. ' + log[-1500:])
            actual_device = int(initialized.group(1) or 0)
            if actual_device != options.vulkan_device:
                raise RuntimeError(f'whisper-cli uruchomił Vulkan {actual_device} zamiast wybranego GPU {options.vulkan_device}.')
            try:
                return json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
            except (OSError, ValueError) as exc:
                raise RuntimeError('whisper-cli nie zwrócił poprawnego wyniku JSON.') from exc
