import io
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.config import Settings
from app.core.audio_processor import chunks, probe, store_upload
from app.core.exporter import export_all, render, timestamp
from app.core.gpu_manager import friendly_error
from app.core.model_manager import ModelManager
from app.core.task_queue import TaskQueue, estimate_eta
from app.database.db import Database
from app.ui.layout import eta_text

FIXTURE = Path(__file__).parent / 'fixtures' / 'jfk.flac'


def test_audio_import_and_incremental_decode(tmp_path):
    with FIXTURE.open('rb') as stream:
        stored, size, duration, warning = store_upload(stream, '../../mowa.flac', 2_000_000, tmp_path)
    assert warning is None
    assert stored.parent == tmp_path
    assert stored.name != 'mowa.flac'
    assert size == FIXTURE.stat().st_size
    assert duration == pytest.approx(11, abs=.1)
    windows = list(chunks(stored, seconds=3))
    assert [offset for offset, _ in windows] == pytest.approx([0, 3, 6, 9])
    assert sum(len(samples) for _, samples in windows) == 11 * 16000
    assert max(len(samples) for _, samples in windows) <= 3 * 16000


def test_upload_rejects_oversize_and_corrupt_files(tmp_path):
    with pytest.raises(ValueError, match='rozmiar'):
        store_upload(io.BytesIO(b'12345'), 'a.mp3', 4, tmp_path)
    with pytest.raises(ValueError, match='Nieobsługiwany'):
        store_upload(io.BytesIO(b'1'), 'a.exe', 10, tmp_path)
    with pytest.raises(ValueError, match='Nie można odczytać'):
        store_upload(io.BytesIO(b'not audio'), 'a.wav', 100, tmp_path)
    assert not list(tmp_path.iterdir())


def test_export_formats_and_polish_text(tmp_path):
    result = {'language': 'pl', 'segments': [{'start': 1.25, 'end': 2.5, 'text': 'Zażółć gęślą jaźń.'},
                                               {'start': 63, 'end': 64, 'text': 'Drugi segment.'}]}
    assert timestamp(3661.234) == '01:01:01.234'
    assert '00:00:01,250 --> 00:00:02,500' in render(result, 'srt')
    assert render(result, 'vtt').startswith('WEBVTT\n\n00:00:01.250')
    assert json.loads(render(result, 'json')) == result
    assert 'Zażółć' in render(result, 'csv')
    export_all(result, tmp_path, 'abc123')
    assert {p.suffix for p in tmp_path.iterdir()} == {'.txt', '.srt', '.vtt', '.json', '.csv'}
    assert 'Zażółć' in (tmp_path / 'abc123.txt').read_text(encoding='utf-8')
    with pytest.raises(ValueError):
        export_all(result, tmp_path, '../escape')


def test_database_queue_order_retry_and_history(tmp_path):
    db = Database(tmp_path / 'jobs.sqlite')
    options = Settings(device='cpu', model='tiny')
    db.save_settings(options)
    assert Database(tmp_path / 'jobs.sqlite').settings().device == 'cpu'
    a = db.add_job('pierwszy.wav', 'a', options)
    b = db.add_job('drugi.wav', 'b', options)
    db.move(b, -1)
    assert db.claim()['id'] == b
    db.cancel(a)
    assert db.get(a)['status'] == 'cancelled'
    db.retry(a)
    assert db.get(a)['status'] == 'waiting'
    db.update(b, status='done', result=json.dumps({'segments': [{'start': 0, 'end': 1, 'text': 'przed'}]}))
    db.edit_segment(b, 0, 'po edycji')
    assert json.loads(Database(tmp_path / 'jobs.sqlite').get(b)['result'])['segments'][0]['text'] == 'po edycji'


def test_interrupted_job_is_recoverable(tmp_path):
    db = Database(tmp_path / 'jobs.sqlite')
    job = db.add_job('nagranie.wav', 'file', Settings())
    db.claim()
    reopened = Database(tmp_path / 'jobs.sqlite')
    assert reopened.get(job)['status'] == 'error'
    reopened.retry(job)
    assert reopened.get(job)['status'] == 'waiting'


def test_live_segments_survive_restart_and_retry_clears_them(tmp_path):
    path = tmp_path / 'jobs.sqlite'
    db = Database(path)
    job = db.add_job('nagranie.wav', 'file', Settings())
    db.claim()
    db.update(job, status='transcribing', progress=.25, eta_seconds=42.5)
    first = {'start': 0.0, 'end': 1.5, 'text': 'Zażółć gęślą jaźń.'}
    second = {'start': 2.0, 'end': 3.0, 'text': 'Drugi segment.'}
    db.append_segment(job, 0, first)
    db.append_segment(job, 1, second)
    assert db.live_segments(job, after=0) == [{'sequence': 1, **second}]
    reopened = Database(path)
    assert reopened.get(job)['status'] == 'error'
    assert reopened.live_segments(job) == [{'sequence': 0, **first}, {'sequence': 1, **second}]
    reopened.retry(job)
    assert reopened.live_segments(job) == []
    assert reopened.get(job)['eta_seconds'] is None
    reopened.claim()
    reopened.append_segment(job, 0, first)
    reopened.update(job, status='cancelled')
    reopened.delete(job)
    assert reopened.get(job) is None and reopened.live_segments(job) == []


def test_database_migrates_existing_history(tmp_path):
    path = tmp_path / 'old.sqlite'
    with sqlite3.connect(path) as conn:
        conn.execute('''CREATE TABLE jobs (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL, status TEXT NOT NULL,
            position INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            options TEXT NOT NULL, progress REAL, duration REAL, elapsed REAL, language TEXT,
            message TEXT DEFAULT '', result TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0)''')
        conn.execute('INSERT INTO jobs(id,name,path,status,position,created_at,updated_at,options,result) VALUES (?,?,?,?,?,?,?,?,?)',
                     ('old', 'wynik.wav', 'file', 'done', 1, '2026', '2026', Settings().model_dump_json(), '{"segments": []}'))
    db = Database(path)
    row = db.get('old')
    assert row['eta_seconds'] is None
    assert row['status'] == 'done' and json.loads(row['result']) == {'segments': []}
    assert db.live_segments('old') == []


def test_eta_uses_measured_processing_speed():
    assert estimate_eta(4, 100, 10) is None
    assert estimate_eta(20, None, 10) is None
    assert estimate_eta(20, 100, .1) is None
    assert estimate_eta(20, 100, 10) == pytest.approx(40)
    assert estimate_eta(100, 100, 10) == 0
    assert 'obliczanie' in eta_text({'status': 'transcribing', 'eta_seconds': None})
    assert eta_text({'status': 'transcribing', 'eta_seconds': 65, 'updated_at': 'invalid'}) == 'ETA: około 00:01:05'
    past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    assert 'szacowany czas minął' in eta_text({'status': 'transcribing', 'eta_seconds': 5, 'updated_at': past})


def test_queue_records_real_file_errors(tmp_path):
    model_dir = tmp_path / 'models' / 'tiny'
    model_dir.mkdir(parents=True)
    for name in ('model.bin', 'config.json', 'tokenizer.json'):
        (model_dir / name).write_bytes(b'invalid')
    bad = tmp_path / 'bad.wav'
    bad.write_bytes(b'not audio')
    db = Database(tmp_path / 'queue.sqlite')
    job = db.add_job('bad.wav', str(bad), Settings(model='tiny', device='cpu', cache_dir=str(model_dir.parent)))
    queue = TaskQueue(db, ModelManager())
    queue.start()
    try:
        deadline = time.monotonic() + 10
        while db.get(job)['status'] not in ('error', 'done') and time.monotonic() < deadline:
            time.sleep(.05)
        state = db.get(job)
        assert state['status'] == 'error'
        assert 'Nie można odczytać' in state['message']
    finally:
        queue.stop()


def test_gpu_errors_are_actionable():
    assert 'VRAM' in friendly_error(RuntimeError('CUDA out of memory'))
    assert 'repair.ps1' in friendly_error(RuntimeError('cudnn64_9.dll missing'))
