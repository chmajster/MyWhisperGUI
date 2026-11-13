from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from app.config import MODEL_REPOS, MODEL_SIZES, ROOT


class DownloadCancelled(Exception):
    """A model download was cancelled by the user."""


class _DownloadRun:
    def __init__(self):
        self.done = threading.Event()
        self.cancel = threading.Event()
        self.error = None


def _worker_command(repo, path, metadata, error):
    return [sys.executable, '-m', 'app.core.model_download_worker', repo, str(path), str(metadata), str(error)]


class ModelManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.states = {}
        self.active = set()
        self.runs = {}
        self.closing = False

    @staticmethod
    def path(name, cache_dir):
        if name not in MODEL_REPOS:
            raise ValueError('Nieznany model.')
        return Path(cache_dir).expanduser().resolve() / name

    def available(self, name, cache_dir):
        directory = self.path(name, cache_dir)
        try:
            return all((directory / f).is_file() and (directory / f).stat().st_size > 0
                       for f in ('model.bin', 'config.json', 'tokenizer.json'))
        except OSError:
            # A downloader can rename or remove a file between the two checks.
            return False

    def is_downloading(self, name, cache_dir):
        key = (str(self.path(name, cache_dir)), name)
        with self.lock:
            return key in self.active

    def details(self, name, cache_dir):
        path = self.path(name, cache_dir)
        size = 0
        try:
            for item in path.rglob('*'):
                try:
                    if item.is_file():
                        size += item.stat().st_size
                except OSError:
                    # Partial files may be moved while progress is refreshed.
                    continue
        except OSError:
            pass
        with self.lock:
            state = dict(self.states.get((str(path), name), {}))
        return {'name': name, 'path': str(path), 'available': self.available(name, cache_dir),
                'size': size, 'estimated': MODEL_SIZES[name], **state}

    def _run_download(self, repo, path, key, run):
        if run.cancel.is_set():
            raise DownloadCancelled('Pobieranie modelu anulowano.')
        with tempfile.TemporaryDirectory(prefix='whysper-model-') as control:
            metadata = Path(control) / 'metadata.json'
            error_file = Path(control) / 'error.txt'
            log_file = Path(control) / 'worker.log'
            env = os.environ.copy()
            env.update(HF_HUB_DISABLE_PROGRESS_BARS='1', PYTHONIOENCODING='utf-8')
            with log_file.open('w', encoding='utf-8') as log:
                process = subprocess.Popen(_worker_command(repo, path, metadata, error_file), cwd=ROOT, env=env,
                                           stdout=subprocess.DEVNULL, stderr=log,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                sizes = None
                try:
                    while process.poll() is None:
                        if run.cancel.is_set():
                            try:
                                process.terminate()
                            except OSError:
                                if process.poll() is None:
                                    raise
                            try:
                                process.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=3)
                            raise DownloadCancelled('Pobieranie modelu anulowano.')
                        if sizes is None and metadata.is_file():
                            try:
                                sizes = json.loads(metadata.read_text(encoding='utf-8'))['sizes']
                            except (OSError, ValueError, KeyError):
                                sizes = None  # The worker may still be writing the file.
                            if sizes is not None:
                                with self.lock:
                                    if not run.cancel.is_set():
                                        self.states[key].update(status='Pobieranie', total=sum(sizes.values()) or None)
                        if sizes is not None:
                            try:
                                complete = sum(min((path / name).stat().st_size, size)
                                               for name, size in sizes.items() if (path / name).is_file())
                                cache = path / '.cache' / 'huggingface' / 'download'
                                partial = sum(item.stat().st_size for item in cache.rglob('*.incomplete') if item.is_file())
                                total = sum(sizes.values()) or None
                                with self.lock:
                                    self.states[key]['downloaded'] = min(complete + partial, total) if total else complete + partial
                            except OSError:
                                pass  # A temporary file may be renamed during the scan.
                        run.cancel.wait(.4)
                    if process.returncode:
                        message = error_file.read_text(encoding='utf-8').strip() if error_file.is_file() else ''
                        raise RuntimeError(message or 'Nie udało się pobrać modelu. Sprawdź połączenie i spróbuj ponownie.')
                    if sizes is None and metadata.is_file():
                        sizes = json.loads(metadata.read_text(encoding='utf-8'))['sizes']
                    return (sum(sizes.values()) or None) if sizes is not None else None
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=3)

    def download(self, name, cache_dir):
        path = self.path(name, cache_dir)
        key = (str(path), name)
        with self.lock:
            if self.closing:
                raise DownloadCancelled('Aplikacja jest zamykana.')
            if key in self.active:
                run = self.runs[key]
                owner = False
            else:
                run = self.runs[key] = _DownloadRun()
                owner = True
                self.active.add(key)
                self.states[key] = {'status': 'Sprawdzanie plików', 'downloaded': 0, 'total': None}
        if not owner:
            run.done.wait()
            if isinstance(run.error, DownloadCancelled):
                raise DownloadCancelled('Pobieranie modelu anulowano.') from run.error
            if run.error is not None:
                raise RuntimeError('Pobieranie modelu nie powiodło się. Sprawdź zakładkę Modele AI.') from run.error
            if self.available(name, cache_dir):
                return path
            raise RuntimeError('Pobieranie modelu nie powiodło się. Sprawdź zakładkę Modele AI.')
        try:
            total = self._run_download(MODEL_REPOS[name], path, key, run)
            if not self.available(name, cache_dir):
                raise RuntimeError('Model jest niekompletny po pobraniu.')
            with self.lock:
                self.states[key].update(status='Pobrano', downloaded=total, total=total)
            return path
        except DownloadCancelled as exc:
            run.error = exc
            with self.lock:
                self.states[key]['status'] = 'Anulowano'
            raise
        except Exception as exc:
            run.error = exc
            with self.lock:
                self.states[key].update(status='Błąd: ' + str(exc))
            raise
        finally:
            with self.lock:
                self.active.discard(key)
                self.runs.pop(key, None)
                run.done.set()

    def cancel_download(self, name, cache_dir):
        key = (str(self.path(name, cache_dir)), name)
        with self.lock:
            run = self.runs.get(key)
            if not run or self.states[key]['status'] not in ('Sprawdzanie plików', 'Pobieranie'):
                return False
            self.states[key]['status'] = 'Anulowanie'
            run.cancel.set()
            return True

    def cancel_all(self):
        with self.lock:
            self.closing = True
            runs = list(self.runs.values())
            for key, run in self.runs.items():
                if self.states[key]['status'] in ('Sprawdzanie plików', 'Pobieranie'):
                    self.states[key]['status'] = 'Anulowanie'
                    run.cancel.set()
        deadline = time.monotonic() + 5
        for run in runs:
            run.done.wait(max(0, deadline - time.monotonic()))

    def start_download(self, name, cache_dir):
        def work():
            try:
                self.download(name, cache_dir)
            except DownloadCancelled:
                pass
            except Exception:
                logging.exception('Błąd pobierania modelu %s', name)
        thread = threading.Thread(target=work, daemon=True)
        thread.start()

    def delete(self, name, cache_dir):
        path = self.path(name, cache_dir)
        with self.lock:
            if (str(path), name) in self.active:
                raise RuntimeError('Nie można usunąć modelu podczas pobierania.')
            if path.exists():
                shutil.rmtree(path)
            self.states.pop((str(path), name), None)
