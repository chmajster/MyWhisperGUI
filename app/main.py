from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import tempfile
import time

from app.config import DATA, LOGS, RECORDINGS, UPLOADS

temporary = DATA / 'tmp'
temporary.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(temporary)
os.environ.setdefault('HF_HOME', str(DATA / 'huggingface'))
logging.basicConfig(level=logging.INFO, handlers=[RotatingFileHandler(LOGS / 'app.log', maxBytes=2_000_000, backupCount=3, encoding='utf-8')],
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response
from nicegui import app, ui

from app.core.exporter import render
from app.core.gpu_manager import configure_cuda_dlls
from app.core.model_manager import ModelManager
from app.core.recorder import Recorder
from app.core.task_queue import TaskQueue
from app.database.db import Database
from app.ui.layout import Interface

configure_cuda_dlls()
db = Database()
models = ModelManager()
queue = TaskQueue(db, models)
recorder = Recorder()


class LocalOriginMiddleware:
    def __init__(self, app):
        self.wrapped = app

    async def __call__(self, scope, receive, send):
        if scope['type'] in ('http', 'websocket'):
            headers = dict(scope.get('headers', []))
            host = headers.get(b'host', b'').decode('latin-1').lower()
            origin = headers.get(b'origin', b'').decode('latin-1').lower()
            allowed = {'127.0.0.1:8080', 'localhost:8080'}
            if host not in allowed or (origin and origin not in {f'http://{name}' for name in allowed}):
                if scope['type'] == 'websocket':
                    await send({'type': 'websocket.close', 'code': 1008})
                else:
                    await Response('Forbidden', status_code=403)(scope, receive, send)
                return
            if scope['type'] == 'http' and scope.get('path', '').startswith('/_nicegui/client/') and '/upload/' in scope['path']:
                try:
                    size = int(headers.get(b'content-length', b'0'))
                except ValueError:
                    size = 0
                if size > db.settings().max_upload_gb * 1024**3 + 1024**2:
                    await Response('Plik przekracza dozwolony rozmiar.', status_code=413)(scope, receive, send)
                    return
        await self.wrapped(scope, receive, send)


app.add_middleware(LocalOriginMiddleware)


def valid_id(job_id):
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise HTTPException(404)
    job = db.get(job_id)
    if not job:
        raise HTTPException(404)
    return job


@app.get('/media/{job_id}')
def media(job_id: str):
    job = valid_id(job_id)
    path = Path(job['path'])
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, content_disposition_type='inline')


@app.get('/recordings/{filename}')
def recording_file(filename: str):
    if not re.fullmatch(r'[0-9a-f]{32}\.wav', filename):
        raise HTTPException(404)
    path = RECORDINGS / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type='audio/wav', content_disposition_type='inline')


@app.get('/export/{job_id}/{format}')
def download(job_id: str, format: str):
    import json
    job = valid_id(job_id)
    if format not in ('txt', 'srt', 'vtt', 'json', 'csv') or not job['result']:
        raise HTTPException(404)
    content = render(json.loads(job['result']), format)
    return Response(content, media_type='application/octet-stream',
                    headers={'Content-Disposition': f'attachment; filename="{job_id}.{format}"'})


@ui.page('/')
def index():
    Interface(db, models, queue, recorder)


def startup():
    # Only disposable temp files and uploads with no history entry are eligible.
    known = {str(Path(j['path']).resolve()) for j in db.list()}
    cutoff = time.time() - 7 * 24 * 3600
    for directory in (temporary, UPLOADS):
        for file in directory.iterdir():
            try:
                if file.is_file() and file.stat().st_mtime < cutoff and str(file.resolve()) not in known:
                    file.unlink()
            except OSError:
                logging.exception('Nie można usunąć starego pliku tymczasowego: %s', file)
    queue.start()


def shutdown():
    if recorder.status()['active']:
        try:
            recorder.stop()
        except Exception:
            logging.exception('Nie można zamknąć nagrywania')
    queue.stop()


app.on_startup(startup)
app.on_shutdown(shutdown)

if __name__ in {'__main__', '__mp_main__'}:
    ui.run(host='127.0.0.1', port=8080, title='WhysperGUI', language='pl', dark=True,
           show=os.environ.get('WHYSPER_NO_BROWSER') != '1', reload=False, favicon='🎙️')
