"""Exercise the Vulkan subprocess protocol, model separation and cancellation."""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from app.config import Settings, VULKAN_MODELS
from app.core import audio_processor, vulkan_engine
from app.core.model_manager import ModelManager
from app.core.task_queue import TaskQueue
from app.core.whisper_engine import Cancelled
from app.database.db import Database


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    script = tmp_path / 'fake_whisper.py'
    script.write_text('''
import json, os, sys, time, wave
args = sys.argv[1:]
def arg(name): return args[args.index(name) + 1]
with wave.open(arg("--file"), "rb") as audio:
    assert audio.getframerate() == 16000 and audio.getnchannels() == 1
if os.environ.get("TEST_VULKAN_SLEEP"):
    time.sleep(20)
if not os.environ.get("TEST_VULKAN_CPU"):
    device = "0" if os.environ.get("TEST_VULKAN_WRONG_DEVICE") else arg("--device")
    print("whisper_backend_init_gpu: using Vulkan" + device + " backend", flush=True)
if os.environ.get("TEST_VULKAN_INIT_FAILURE"):
    print("whisper_backend_init_gpu: failed to initialize Vulkan0 backend", flush=True)
with open(arg("--output-file") + ".json", "w", encoding="utf-8") as output:
    json.dump({"result": {"language": "pl"}, "transcription": [
        {"offsets": {"from": 100, "to": 400}, "text": " Witaj"}]}, output)
''', encoding='utf-8')
    real_popen = subprocess.Popen
    commands = []

    def run_fake(command, **kwargs):
        commands.append(command)
        return real_popen([sys.executable, str(script), *command[1:]], **kwargs)

    monkeypatch.setattr(vulkan_engine, 'whisper_cli', lambda: 'fake-whisper-cli')
    monkeypatch.setattr(vulkan_engine.subprocess, 'Popen', run_fake)
    return commands


def model_and_options(tmp_path, **overrides):
    manager = ModelManager()
    options = Settings(model='tiny', device='vulkan', cache_dir=str(tmp_path), vad=False, **overrides)
    path = manager.path('tiny', tmp_path, 'vulkan')
    path.mkdir(parents=True)
    (path / VULKAN_MODELS['tiny']).write_bytes(b'model')
    return manager, options


def test_model_formats_are_separate(tmp_path):
    manager, options = model_and_options(tmp_path)
    assert manager.available('tiny', tmp_path, 'vulkan')
    assert not manager.available('tiny', tmp_path)
    assert manager.path('tiny', tmp_path) != manager.path('tiny', tmp_path, 'vulkan')
    assert 'ggml-large-v3-turbo.bin' == VULKAN_MODELS['turbo']


def test_vad_ranges_use_original_sample_positions():
    options = Settings(device='vulkan', model='tiny', vad=True)
    source = Path(__file__).parent / 'fixtures' / 'jfk.flac'
    samples = next(audio_processor.chunks(source))[1]
    ranges = vulkan_engine._speech_ranges(samples, options)
    assert ranges and all(0 <= start < end <= len(samples) for start, end in ranges)
    assert all(first[1] <= second[0] for first, second in zip(ranges, ranges[1:]))
    assert vulkan_engine._speech_ranges(np.zeros(16000, dtype=np.float32), options) == []


def test_vulkan_download_uses_ggml_repository_and_own_directory(tmp_path, monkeypatch):
    manager = ModelManager()
    calls = []

    def download(repo, path, key, run):
        calls.append((repo, path))
        path.mkdir(parents=True)
        (path / VULKAN_MODELS[path.name]).write_bytes(b'ggml')
        return 4

    monkeypatch.setattr(manager, '_run_download', download)
    result = manager.download('tiny', tmp_path, 'vulkan')
    assert calls[0][0] == 'ggerganov/whisper.cpp'
    assert result == tmp_path / 'whisper-cpp' / 'tiny'
    assert manager.details('tiny', tmp_path, 'vulkan')['available']
    assert not manager.available('tiny', tmp_path)


def test_queue_transcribes_and_exports_with_vulkan(tmp_path, fake_cli):
    manager, options = model_and_options(tmp_path, output_dir=str(tmp_path / 'exports'))
    db = Database(tmp_path / 'history.sqlite3')
    source = Path(__file__).parent / 'fixtures' / 'jfk.flac'
    job_id = db.add_job('recording.flac', str(source), options)
    queue = TaskQueue(db, manager)
    queue._run(db.claim(), None)
    job = db.get(job_id)
    assert job['status'] == 'done', job['message']
    result = json.loads(job['result'])
    assert result['device'] == 'vulkan' and result['segments'][0]['text'] == 'Witaj'
    assert (tmp_path / 'exports' / f'{job_id}.txt').read_text(encoding='utf-8').strip() == 'Witaj'


def test_vulkan_transcription_maps_vad_times_and_emits_segments(tmp_path, monkeypatch, fake_cli):
    manager, options = model_and_options(tmp_path, word_timestamps=True, vulkan_device=2)
    monkeypatch.setattr(vulkan_engine, '_speech_ranges', lambda samples, options: [(8000, 24000)])
    engine = vulkan_engine.VulkanEngine(manager)
    emitted, progress = [], []
    result = engine._transcribe([(10.0, np.zeros(32000, dtype=np.float32))], options, progress.append,
                                lambda: False, emitted.append)
    segment = result['segments'][0]
    assert result['device'] == 'vulkan' and result['language'] == 'pl'
    assert segment['start'] == pytest.approx(10.6) and segment['end'] == pytest.approx(10.9)
    assert segment['words'][0]['word'] == 'Witaj' and emitted == result['segments']
    assert progress[-1] == 12
    command = fake_cli[0]
    assert command[command.index('--device') + 1] == '2'
    assert '--split-on-word' in command and '--no-fallback' in command


def test_vulkan_rejects_silent_cpu_fallback(tmp_path, monkeypatch, fake_cli):
    manager, options = model_and_options(tmp_path)
    monkeypatch.setenv('TEST_VULKAN_CPU', '1')
    with pytest.raises(RuntimeError, match='nie potwierdził'):
        vulkan_engine.VulkanEngine(manager).transcribe_samples(np.zeros(16000), options, lambda: False)


def test_vulkan_rejects_failed_initialization_even_if_using_was_logged(tmp_path, monkeypatch, fake_cli):
    manager, options = model_and_options(tmp_path)
    monkeypatch.setenv('TEST_VULKAN_INIT_FAILURE', '1')
    with pytest.raises(RuntimeError, match='nie potwierdził'):
        vulkan_engine.VulkanEngine(manager).transcribe_samples(np.zeros(16000), options, lambda: False)


def test_vulkan_rejects_a_different_gpu(tmp_path, monkeypatch, fake_cli):
    manager, options = model_and_options(tmp_path, vulkan_device=1)
    monkeypatch.setenv('TEST_VULKAN_WRONG_DEVICE', '1')
    with pytest.raises(RuntimeError, match='zamiast wybranego GPU'):
        vulkan_engine.VulkanEngine(manager).transcribe_samples(np.zeros(16000), options, lambda: False)


def test_vulkan_cancellation_stops_subprocess(tmp_path, monkeypatch, fake_cli):
    manager, options = model_and_options(tmp_path)
    monkeypatch.setenv('TEST_VULKAN_SLEEP', '1')
    start = time.monotonic()
    with pytest.raises(Cancelled):
        vulkan_engine.VulkanEngine(manager).transcribe_samples(np.zeros(16000), options,
                                                               lambda: time.monotonic() - start > .3)
    assert time.monotonic() - start < 5


def test_vulkan_rejects_unsupported_advanced_options(tmp_path, fake_cli):
    manager, options = model_and_options(tmp_path, hotwords='nazwisko')
    with pytest.raises(ValueError, match='tylko w faster-whisper'):
        vulkan_engine.VulkanEngine(manager).prepare(options)
