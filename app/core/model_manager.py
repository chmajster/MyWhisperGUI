from __future__ import annotations

from pathlib import Path
import shutil
import threading

from app.config import MODEL_REPOS, MODEL_SIZES


class ModelManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.states = {}
        self.active = set()
        self.finished = {}

    @staticmethod
    def path(name, cache_dir):
        if name not in MODEL_REPOS:
            raise ValueError('Nieznany model.')
        return Path(cache_dir).expanduser().resolve() / name

    def available(self, name, cache_dir):
        directory = self.path(name, cache_dir)
        return all((directory / f).is_file() and (directory / f).stat().st_size > 0
                   for f in ('model.bin', 'config.json', 'tokenizer.json'))

    def details(self, name, cache_dir):
        path = self.path(name, cache_dir)
        size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.exists() else 0
        with self.lock:
            state = dict(self.states.get((str(path), name), {}))
        return {'name': name, 'path': str(path), 'available': self.available(name, cache_dir),
                'size': size, 'estimated': MODEL_SIZES[name], **state}

    def download(self, name, cache_dir):
        from huggingface_hub import HfApi, snapshot_download
        path = self.path(name, cache_dir)
        key = (str(path), name)
        with self.lock:
            if key in self.active:
                other = self.finished[key]
                owner = False
            else:
                other = self.finished[key] = threading.Event()
                owner = True
                self.active.add(key)
                self.states[key] = {'status': 'Sprawdzanie plików', 'downloaded': 0, 'total': None}
        if not owner:
            other.wait()
            if self.available(name, cache_dir):
                return path
            raise RuntimeError('Pobieranie modelu nie powiodło się. Sprawdź zakładkę Modele AI.')
        stop = threading.Event()
        observer = None
        try:
            repo = MODEL_REPOS[name]
            info = HfApi().model_info(repo, files_metadata=True)
            allowed = {'model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json',
                       'vocabulary.txt', 'preprocessor_config.json'}
            sizes = {s.rfilename: s.size for s in info.siblings if s.rfilename in allowed and s.size is not None}
            total = sum(sizes.values()) or None
            with self.lock:
                self.states[key].update(status='Pobieranie', total=total)

            def monitor():
                while not stop.wait(0.4):
                    try:
                        complete = sum(min((path / n).stat().st_size, size) for n, size in sizes.items() if (path / n).is_file())
                        cache = path / '.cache' / 'huggingface' / 'download'
                        partial = sum(p.stat().st_size for p in cache.rglob('*.incomplete') if p.is_file()) if cache.exists() else 0
                        with self.lock:
                            self.states[key]['downloaded'] = min(complete + partial, total) if total else complete + partial
                    except OSError:
                        # A temporary file may be renamed while its size is read.
                        continue
            observer = threading.Thread(target=monitor, daemon=True)
            observer.start()
            path.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo, local_dir=path, allow_patterns=list(allowed), max_workers=1)
            if not self.available(name, cache_dir):
                raise RuntimeError('Model jest niekompletny po pobraniu.')
            with self.lock:
                self.states[key].update(status='Pobrano', downloaded=total, total=total)
            return path
        except Exception as exc:
            with self.lock:
                self.states[key].update(status='Błąd: ' + str(exc))
            raise
        finally:
            stop.set()
            if observer:
                observer.join(timeout=2)
            with self.lock:
                self.active.discard(key)
                other.set()

    def start_download(self, name, cache_dir):
        def work():
            try:
                self.download(name, cache_dir)
            except Exception:
                import logging
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
