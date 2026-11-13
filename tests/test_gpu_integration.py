import json
import os
from pathlib import Path
import time
import wave

import pytest
import av
import numpy as np

from app.config import ROOT, Settings
from app.core.audio_processor import chunks
from app.core.gpu_manager import gpu_stats, inference_test
from app.core.model_manager import ModelManager
from app.core.task_queue import TaskQueue
from app.database.db import Database

FIXTURE = Path(__file__).parent / 'fixtures' / 'jfk.flac'


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get('WHYSPER_GPU_TEST') != '1', reason='Set WHYSPER_GPU_TEST=1 to run actual RTX 5070 Ti inference')
def test_rtx_5070_ti_transcribes_and_persists(tmp_path):
    assert 'RTX 5070 Ti' in gpu_stats().get('name', ''), 'This test specifically requires RTX 5070 Ti.'
    model = ROOT / 'models' / 'tiny'
    assert ModelManager().available('tiny', model.parent), 'Download tiny before running the test.'
    check = inference_test(model, FIXTURE)
    assert check.get('ok'), check
    assert check['device'] == 'cuda'
    assert check['compute_type'] == 'float16'
    assert check['language'] == 'en'
    assert 'country' in check['text'].lower()

    # Several real audio windows leave time to observe partial segments before completion.
    sample = next(chunks(FIXTURE))[1]
    pcm = (np.tile(sample, 8) * 32767).astype('<i2').tobytes()
    longer_audio = tmp_path / 'jfk-long.wav'
    with wave.open(str(longer_audio), 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(pcm)
    db = Database(tmp_path / 'history.sqlite')
    options = Settings(model='tiny', device='cuda', compute_type='float16', language='', beam_size=1,
                       vad=False, cache_dir=str(model.parent), output_dir=str(tmp_path / 'exports'))
    job = db.add_job('jfk-long.wav', str(longer_audio), options)
    tasks = TaskQueue(db, ModelManager())
    tasks.start()
    try:
        deadline = time.monotonic() + 180
        saw_live = False
        saw_eta = False
        while time.monotonic() < deadline:
            current = db.get(job)
            if current['status'] == 'transcribing' and db.live_segments(job):
                saw_live = True
            if current['status'] == 'transcribing' and current['eta_seconds'] is not None and current['eta_seconds'] > 0:
                saw_eta = True
            if current['status'] in ('done', 'error'):
                break
            time.sleep(.03)
        row = db.get(job)
        assert row['status'] == 'done', row['message']
        result = json.loads(row['result'])
        assert saw_live, 'No recognized segment was available while transcription was still running.'
        assert saw_eta, 'No measured ETA was available while transcription was still running.'
        assert [{k: v for k, v in segment.items() if k != 'sequence'} for segment in db.live_segments(job)] == result['segments']
        assert result['device'] == 'cuda'
        assert result['language'] == 'en'
        assert result['segments'] and 'country' in ' '.join(s['text'] for s in result['segments']).lower()
        assert 0 <= result['segments'][0]['start'] < result['segments'][0]['end'] <= row['duration'] + .1
        assert row['progress'] == 1
        assert (tmp_path / 'exports' / f'{job}.txt').is_file()
        assert (tmp_path / 'exports' / f'{job}.srt').is_file()
        assert Database(tmp_path / 'history.sqlite').get(job)['status'] == 'done'
    finally:
        tasks.stop()


@pytest.mark.large_gpu
@pytest.mark.skipif(os.environ.get('WHYSPER_LARGE_TEST') != '1', reason='Set WHYSPER_LARGE_TEST=1 to test large-v3 on RTX 5070 Ti')
def test_large_v3_polish_mp3_end_to_end(tmp_path):
    assert 'RTX 5070 Ti' in gpu_stats().get('name', '')
    manager = ModelManager()
    assert manager.available('large-v3', ROOT / 'models'), 'Download large-v3 before running the test.'
    source = Path(__file__).parent / 'fixtures' / 'polish.wav'
    mp3 = tmp_path / 'polskie-nagranie.mp3'
    with av.open(str(source)) as input_file, av.open(str(mp3), mode='w') as output_file:
        stream = output_file.add_stream('libmp3lame', rate=44100)
        stream.layout = 'mono'
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=44100)
        for frame in input_file.decode(audio=0):
            for converted in resampler.resample(frame):
                for packet in stream.encode(converted):
                    output_file.mux(packet)
        for converted in resampler.resample(None):
            for packet in stream.encode(converted):
                output_file.mux(packet)
        for packet in stream.encode(None):
            output_file.mux(packet)

    db = Database(tmp_path / 'history.sqlite')
    options = Settings(model='large-v3', language='pl', device='cuda', compute_type='float16', beam_size=5,
                       vad=False, cache_dir=str(ROOT / 'models'), output_dir=str(tmp_path / 'exports'))
    job = db.add_job(mp3.name, str(mp3), options)
    tasks = TaskQueue(db, manager)
    tasks.start()
    try:
        deadline = time.monotonic() + 240
        while db.get(job)['status'] not in ('done', 'error') and time.monotonic() < deadline:
            time.sleep(.2)
        row = db.get(job)
        assert row['status'] == 'done', row['message']
        result = json.loads(row['result'])
        text = ' '.join(s['text'] for s in result['segments']).lower()
        assert result['device'] == 'cuda' and result['model'] == 'large-v3'
        assert result['language'] == 'pl'
        assert 'oczy' in text and 'śmiech' in text, text
        assert row['progress'] == 1
        assert 'oczy' in (tmp_path / 'exports' / f'{job}.txt').read_text(encoding='utf-8').lower()
        assert '-->' in (tmp_path / 'exports' / f'{job}.srt').read_text(encoding='utf-8')
        assert Database(tmp_path / 'history.sqlite').get(job)['status'] == 'done'
    finally:
        tasks.stop()
