import html
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid

import av
import numpy as np
import pytest
import requests
import socketio

from app.config import DATA, DB_PATH, UPLOADS, Settings
from app.core import audio_processor


@pytest.fixture
def video(tmp_path):
    path = tmp_path / 'oryginal.mp4'
    with av.open(str(path), mode='w') as output:
        video_stream = output.add_stream('mpeg4', rate=5)
        video_stream.width, video_stream.height, video_stream.pix_fmt = 64, 64, 'yuv420p'
        audio_stream = output.add_stream('aac', rate=48000)
        audio_stream.layout = 'stereo'
        for index in range(5):
            pixels = np.full((64, 64, 3), index * 40, dtype=np.uint8)
            for packet in video_stream.encode(av.VideoFrame.from_ndarray(pixels, format='rgb24')):
                output.mux(packet)
        samples = np.arange(48000) * 2 * np.pi / 48000
        stereo = np.stack((np.sin(samples * 440), np.sin(samples * 880)))
        frame = av.AudioFrame.from_ndarray((stereo * 20000).astype(np.int16), format='s16p', layout='stereo')
        frame.sample_rate = 48000
        for packet in audio_stream.encode(frame):
            output.mux(packet)
        for packet in audio_stream.encode(None):
            output.mux(packet)
        for packet in video_stream.encode(None):
            output.mux(packet)
    return path


@pytest.mark.parametrize('name', ['film.mp4', 'film.mkv', 'film.mov', 'film.avi', 'film.webm'])
def test_video_upload_keeps_verified_audio_and_removes_only_the_copy(tmp_path, video, name):
    uploads = tmp_path / 'uploads'
    original = video.read_bytes()
    with video.open('rb') as source:
        stored, size, duration, warning = audio_processor.store_upload(source, name, 1_000_000, uploads)

    assert warning is None
    assert video.read_bytes() == original
    assert stored.suffix == '.flac'
    assert list(uploads.iterdir()) == [stored]
    assert size == stored.stat().st_size
    expected = np.concatenate([samples for _, samples in audio_processor.chunks(video)])
    actual = np.concatenate([samples for _, samples in audio_processor.chunks(stored)])
    np.testing.assert_array_equal(actual, expected)
    assert duration == pytest.approx(len(actual) / 16000)
    with av.open(str(stored)) as audio:
        assert len(audio.streams.audio) == 1
        assert not audio.streams.video


@pytest.mark.parametrize('failure', ['encode', 'verify'])
def test_failed_conversion_preserves_uploaded_video_and_cleans_partial_audio(tmp_path, video, monkeypatch, failure):
    uploads = tmp_path / 'uploads'
    original_open = av.open

    def failing_open(file, *args, **kwargs):
        if failure == 'encode' and kwargs.get('mode') == 'w' and kwargs.get('format') == 'flac':
            raise OSError('błąd zapisu')
        if failure == 'verify' and isinstance(file, str) and file.endswith('.flac') and kwargs.get('mode') == 'r':
            raise OSError('błąd weryfikacji')
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(audio_processor.av, 'open', failing_open)
    with video.open('rb') as source:
        stored, size, duration, warning = audio_processor.store_upload(source, video.name, 1_000_000, uploads)

    assert stored.suffix == '.mp4'
    assert stored.read_bytes() == video.read_bytes()
    assert list(uploads.iterdir()) == [stored]
    assert size == video.stat().st_size
    assert duration > 0
    assert 'Zachowano lokalną kopię wideo' in warning


def test_failed_video_cleanup_keeps_audio_and_reports_warning(tmp_path, video, monkeypatch):
    uploads = tmp_path / 'uploads'
    original_unlink = Path.unlink

    def failing_unlink(path, *args, **kwargs):
        if path.parent == uploads and path.suffix == '.mp4':
            raise PermissionError('plik w użyciu')
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', failing_unlink)
    with video.open('rb') as source:
        stored, _, _, warning = audio_processor.store_upload(source, video.name, 1_000_000, uploads)

    assert stored.suffix == '.flac'
    assert len(list(audio_processor.chunks(stored))) > 0
    assert stored.with_suffix('.mp4').read_bytes() == video.read_bytes()
    assert 'nie udało się usunąć lokalnej kopii wideo' in warning


def test_upload_collision_does_not_remove_existing_file(tmp_path, video, monkeypatch):
    fixed_id = uuid.UUID('01234567-89ab-cdef-0123-456789abcdef')
    existing = tmp_path / f'{fixed_id.hex}.mp4'
    existing.write_bytes(b'existing file')
    monkeypatch.setattr(audio_processor.uuid, 'uuid4', lambda: fixed_id)

    with video.open('rb') as source, pytest.raises(FileExistsError):
        audio_processor.store_upload(source, video.name, 1_000_000, tmp_path)
    assert existing.read_bytes() == b'existing file'


@pytest.mark.ui
@pytest.mark.skipif(os.environ.get('WHYSPER_UI_TEST') != '1', reason='Set WHYSPER_UI_TEST=1 with the app running')
def test_video_upload_and_audio_player_over_localhost(video):
    # Force the web server to spool the request to disk, as it does for large videos.
    with video.open('ab') as source:
        source.write(b'\0' * (1024 * 1024 + 1))
    page = requests.get('http://127.0.0.1:8080/', timeout=5)
    page.raise_for_status()
    elements = json.loads(html.unescape(re.search(r'parseElements\(String.raw`(.*?)`\)', page.text, re.S).group(1)))
    client_id = re.search(r"'client_id': '([\w-]+)'", page.text).group(1)
    changed = threading.Condition()
    socket = socketio.Client(reconnection=False)
    job_id = uuid.uuid4().hex
    upload_name = f'video-test-{job_id}.mp4'
    before = set(UPLOADS.iterdir())
    temporary = DATA / 'tmp'
    temporary_before = set(temporary.iterdir())
    saved_audio = None

    @socket.on('update')
    def update(data):
        with changed:
            for key, value in data.items():
                if key == '_id':
                    continue
                if value is None:
                    elements.pop(str(key), None)
                else:
                    elements[str(key)] = value
            changed.notify_all()

    def click(element_id, item):
        listener = next(event['listener_id'] for event in item['events'] if event['type'] == 'click')
        socket.emit('event', {'id': int(element_id), 'client_id': client_id, 'listener_id': listener, 'args': []})

    def wait_for(predicate, message):
        deadline = time.monotonic() + 10
        while not predicate() and time.monotonic() < deadline:
            changed.wait(.1)
        assert predicate(), message

    def descendants(root):
        result, todo = [], [root]
        while todo:
            item = elements.get(str(todo.pop()))
            if item:
                result.append(item)
                todo.extend(item.get('children', []))
        return result

    try:
        socket.connect('http://127.0.0.1:8080?client_id=' + client_id + '&next_message_id=0',
                       socketio_path='/_nicegui_ws/socket.io', transports=['polling'])
        assert socket.call('handshake', {'client_id': client_id, 'document_id': str(uuid.uuid4()),
                                        'tab_id': str(uuid.uuid4()), 'next_message_id': 0}, timeout=5)
        with changed:
            nav = next((key, item) for key, item in elements.items() if item and item.get('props', {}).get('label') == 'Transkrypcja')
            click(*nav)
            wait_for(lambda: any(item and item.get('tag') == 'nicegui-upload' for item in elements.values()),
                     'Formularz importu nie został wyświetlony')
            upload = next(item for item in elements.values() if item and item.get('tag') == 'nicegui-upload')
        with video.open('rb') as source:
            response = requests.post('http://127.0.0.1:8080' + upload['props']['url'],
                                     files={'file': (upload_name, source, 'video/mp4')}, timeout=20)
        response.raise_for_status()
        with changed:
            wait_for(lambda: any(item and upload_name in item.get('text', '') for item in elements.values()),
                     'Zaimportowane wideo nie zostało dodane do formularza')

        created = set(UPLOADS.iterdir()) - before
        assert len(created) == 1
        saved_audio = created.pop()
        assert saved_audio.suffix == '.flac'
        assert video.exists()
        assert not (set(temporary.iterdir()) - temporary_before)

        result = {'model': 'tiny', 'device': 'cpu', 'language': 'pl', 'segments': []}
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''INSERT INTO jobs(id,name,path,status,position,created_at,updated_at,options,result)
                            VALUES (?,?,?,?,?,?,?,?,?)''',
                         (job_id, upload_name, str(saved_audio), 'done', 999999, '2026', '2026',
                          Settings().model_dump_json(), json.dumps(result)))
        with changed:
            nav = next((key, item) for key, item in elements.items() if item and item.get('props', {}).get('label') == 'Historia')
            click(*nav)
            wait_for(lambda: any(item and item.get('text') == upload_name for item in elements.values()),
                     'Wynik nie został wyświetlony w historii')
            row = next(key for key, item in elements.items() if item and 'job-row' in item.get('class', [])
                       and any(child.get('text') == upload_name for child in descendants(key)))
            button = next((key, item) for key, item in elements.items() if item in descendants(row)
                          and item.get('props', {}).get('label') == 'Otwórz')
            click(*button)
            wait_for(lambda: any(item and item.get('tag') == 'nicegui-audio'
                                 and item.get('props', {}).get('src') == f'/media/{job_id}'
                                 for item in elements.values()), 'Edytor nie wyświetlił odtwarzacza audio')
        response = requests.get(f'http://127.0.0.1:8080/media/{job_id}', timeout=5)
        response.raise_for_status()
        assert response.content.startswith(b'fLaC')
    finally:
        if socket.connected:
            socket.disconnect()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('DELETE FROM jobs WHERE id=?', (job_id,))
        if saved_audio and saved_audio.parent.resolve() == UPLOADS.resolve():
            saved_audio.unlink(missing_ok=True)
