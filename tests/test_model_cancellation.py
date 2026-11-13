import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.core import model_download_worker, model_manager
from app.core.model_manager import DownloadCancelled, ModelManager
from app.core.task_queue import TaskQueue
from app.database.db import Database


def wait_until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    assert predicate(), 'Timed out while waiting for the download'


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    script = tmp_path / 'fake_download.py'
    script.write_text('''
import json, sys, time
from pathlib import Path
repo, target, metadata, error, started, finish, phase = sys.argv[1:]
target, metadata = Path(target), Path(metadata)
target.mkdir(parents=True, exist_ok=True)
partial = target / '.cache' / 'huggingface' / 'download' / 'model.incomplete'
partial.parent.mkdir(parents=True, exist_ok=True)
partial.write_bytes(b'partial')
Path(started).write_text('started')
if phase == 'download' or Path(finish).exists():
    metadata.write_text(json.dumps({'sizes': {'model.bin': 100}}))
if Path(finish).exists():
    for name in ('model.bin', 'config.json', 'tokenizer.json'):
        (target / name).write_bytes(b'complete')
else:
    while True:
        time.sleep(.05)
''', encoding='utf-8')
    started = tmp_path / 'started.flag'
    finish = tmp_path / 'finish.flag'
    phase = {'value': 'download'}

    def command(repo, path, metadata, error):
        return [sys.executable, str(script), repo, str(path), str(metadata), str(error),
                str(started), str(finish), phase['value']]

    monkeypatch.setattr(model_manager, '_worker_command', command)
    return started, finish, phase


@pytest.mark.parametrize('phase', ['checking', 'download'])
def test_cancel_stops_worker_preserves_partial_files_and_can_resume(tmp_path, fake_worker, phase):
    started, finish, settings = fake_worker
    settings['value'] = phase
    cache = tmp_path / 'models'
    manager = ModelManager()
    failures = []

    def download():
        try:
            manager.download('tiny', cache)
        except Exception as exc:
            failures.append(exc)

    thread = threading.Thread(target=download)
    thread.start()
    try:
        wait_until(started.is_file)
        expected = 'Pobieranie' if phase == 'download' else 'Sprawdzanie plików'
        wait_until(lambda: manager.details('tiny', cache).get('status') == expected)
        assert manager.cancel_download('tiny', cache)
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(failures) == 1 and isinstance(failures[0], DownloadCancelled)
        assert manager.details('tiny', cache)['status'] == 'Anulowano'
        assert (cache / 'tiny' / '.cache' / 'huggingface' / 'download' / 'model.incomplete').read_bytes() == b'partial'
        assert not manager.cancel_download('tiny', cache)

        finish.write_text('continue', encoding='utf-8')
        assert manager.download('tiny', cache) == (cache / 'tiny').resolve()
        assert manager.available('tiny', cache)
        assert manager.details('tiny', cache)['status'] == 'Pobrano'
    finally:
        manager.cancel_all()
        thread.join(timeout=5)


def test_cancel_notifies_all_waiters_even_when_required_files_already_exist(tmp_path, fake_worker, monkeypatch):
    started, _, _ = fake_worker
    cache = tmp_path / 'models'
    manager = ModelManager()
    failures = []

    def download():
        try:
            manager.download('tiny', cache)
        except Exception as exc:
            failures.append(exc)

    owner = threading.Thread(target=download)
    waiter = threading.Thread(target=download)
    owner.start()
    try:
        wait_until(started.is_file)
        key = (str((cache / 'tiny').resolve()), 'tiny')
        run = manager.runs[key]
        entered = threading.Event()
        original_wait = run.done.wait

        def waiter_wait(timeout=None):
            entered.set()
            return original_wait(timeout)

        monkeypatch.setattr(run.done, 'wait', waiter_wait)
        waiter.start()
        assert entered.wait(5)
        for filename in ('model.bin', 'config.json', 'tokenizer.json'):
            (cache / 'tiny' / filename).write_bytes(b'present')
        assert manager.available('tiny', cache)
        assert manager.cancel_download('tiny', cache)
        owner.join(timeout=5)
        waiter.join(timeout=5)
        assert not owner.is_alive() and not waiter.is_alive()
        assert len(failures) == 2 and all(isinstance(exc, DownloadCancelled) for exc in failures)
    finally:
        manager.cancel_all()
        owner.join(timeout=5)
        if waiter.ident is not None:
            waiter.join(timeout=5)


def test_stopping_manager_cancels_downloads_and_rejects_new_ones(tmp_path, fake_worker):
    started, _, _ = fake_worker
    manager = ModelManager()
    manager.start_download('tiny', tmp_path)
    wait_until(started.is_file)
    manager.cancel_all()
    assert not manager.active
    assert manager.details('tiny', tmp_path)['status'] == 'Anulowano'
    with pytest.raises(DownloadCancelled):
        manager.download('base', tmp_path)


def test_worker_reports_sizes_and_downloads_only_required_files(tmp_path, monkeypatch):
    class FakeApi:
        def model_info(self, repo, files_metadata):
            assert repo == 'org/model' and files_metadata
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename='model.bin', size=100),
                                             SimpleNamespace(rfilename='other.txt', size=20)])

    calls = []
    monkeypatch.setattr(model_download_worker, 'HfApi', FakeApi)
    monkeypatch.setattr(model_download_worker, 'snapshot_download', lambda *args, **kwargs: calls.append((args, kwargs)))
    metadata = tmp_path / 'metadata.json'
    model_download_worker.download('org/model', str(tmp_path / 'model'), metadata, tmp_path / 'error.txt')
    assert json.loads(metadata.read_text(encoding='utf-8')) == {'sizes': {'model.bin': 100}}
    assert calls[0][0] == ('org/model',)
    assert set(calls[0][1]['allow_patterns']) == model_download_worker.ALLOWED_FILES
    assert calls[0][1]['max_workers'] == 1


def test_worker_reports_download_errors(tmp_path, monkeypatch):
    class OfflineApi:
        def model_info(self, repo, files_metadata):
            raise RuntimeError('Brak połączenia testowego')

    monkeypatch.setattr(model_download_worker, 'HfApi', OfflineApi)
    error = tmp_path / 'error.txt'
    with pytest.raises(RuntimeError, match='Brak połączenia testowego'):
        model_download_worker.download('org/model', str(tmp_path / 'model'), tmp_path / 'metadata.json', error)
    assert error.read_text(encoding='utf-8') == 'Brak połączenia testowego'


def test_failed_worker_reports_error_to_the_ui(tmp_path, monkeypatch):
    script = tmp_path / 'fail.py'
    script.write_text('from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text("offline")\nsys.exit(1)\n', encoding='utf-8')
    monkeypatch.setattr(model_manager, '_worker_command',
                        lambda repo, path, metadata, error: [sys.executable, str(script), str(error)])
    manager = ModelManager()
    with pytest.raises(RuntimeError, match='offline'):
        manager.download('tiny', tmp_path)
    assert manager.details('tiny', tmp_path)['status'] == 'Błąd: offline'
    assert not manager.active


def test_task_is_cancelled_when_its_model_download_is_cancelled(tmp_path):
    class CancelledModels:
        def available(self, name, cache_dir):
            return False

        def download(self, name, cache_dir):
            raise DownloadCancelled('Pobieranie modelu anulowano.')

    db = Database(tmp_path / 'jobs.sqlite')
    job_id = db.add_job('audio.wav', str(tmp_path / 'audio.wav'), Settings(model='tiny', device='cpu'))
    queue = TaskQueue(db, CancelledModels())
    queue._run(db.claim(), None)
    job = db.get(job_id)
    assert job['status'] == 'cancelled'
    assert 'Pobieranie modelu anulowano' in job['message']
