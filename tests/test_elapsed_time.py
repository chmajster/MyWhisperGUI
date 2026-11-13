import threading
from pathlib import Path

import pytest

from app.config import Settings
from app.core import task_queue
from app.database.db import Database


def test_elapsed_time_advances_before_progress_and_stops_when_job_finishes(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(task_queue.time, 'monotonic', lambda: clock[0])
    entered, finish = threading.Event(), threading.Event()

    class ReadyModels:
        def available(self, name, cache_dir):
            return True

    class SlowEngine:
        def transcribe(self, path, options, progress, cancelled, on_segment, on_started):
            on_started()
            entered.set()
            assert finish.wait(5)
            return {'language': 'pl', 'segments': []}

    db = Database(tmp_path / 'jobs.sqlite')
    source = Path(__file__).parent / 'fixtures' / 'polish.wav'
    options = Settings(model='tiny', device='cpu', cache_dir=str(tmp_path), output_dir=str(tmp_path / 'exports'))
    job_id = db.add_job('mowa.wav', str(source), options)
    queue = task_queue.TaskQueue(db, ReadyModels())
    assert queue.elapsed(db.get(job_id)) is None
    job = db.claim()
    worker = threading.Thread(target=queue._run, args=(job, SlowEngine()))
    worker.start()
    try:
        assert entered.wait(3)
        assert db.get(job_id)['elapsed'] is None  # No progress callback has occurred.
        clock[0] = 111.5
        assert queue.elapsed(db.get(job_id)) == pytest.approx(11.5)
        clock[0] = 161.0
        assert queue.elapsed(db.get(job_id)) == pytest.approx(61.0)
        finish.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
        completed = db.get(job_id)
        assert completed['status'] == 'done'
        assert completed['elapsed'] == pytest.approx(61.0)
        clock[0] = 500.0
        assert queue.elapsed(completed) == pytest.approx(61.0)
    finally:
        finish.set()
        worker.join(timeout=5)


def test_elapsed_time_advances_while_model_downloads_and_stops_on_cancel(tmp_path, monkeypatch):
    clock = [200.0]
    monkeypatch.setattr(task_queue.time, 'monotonic', lambda: clock[0])
    entered, finish = threading.Event(), threading.Event()

    class SlowModels:
        def available(self, name, cache_dir):
            return False

        def download(self, name, cache_dir):
            entered.set()
            assert finish.wait(5)

    db = Database(tmp_path / 'jobs.sqlite')
    options = Settings(model='tiny', device='cpu', cache_dir=str(tmp_path), output_dir=str(tmp_path / 'exports'))
    job_id = db.add_job('mowa.wav', 'unused.wav', options)
    queue = task_queue.TaskQueue(db, SlowModels())
    worker = threading.Thread(target=queue._run, args=(db.claim(), None))
    worker.start()
    try:
        assert entered.wait(3)
        assert db.get(job_id)['status'] == 'downloading'
        clock[0] = 230.0
        assert queue.elapsed(db.get(job_id)) == pytest.approx(30.0)
        db.cancel(job_id)
        clock[0] = 235.0
        finish.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
        cancelled = db.get(job_id)
        assert cancelled['status'] == 'cancelled'
        clock[0] = 500.0
        assert queue.elapsed(cancelled) == pytest.approx(35.0)
    finally:
        finish.set()
        worker.join(timeout=5)
