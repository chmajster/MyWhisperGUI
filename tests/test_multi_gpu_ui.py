"""Integration test for a server started with two simulated CUDA devices."""
import html
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid

import pytest
import requests
import socketio

from app.config import DB_PATH, UPLOADS, Settings


GPU_B = 'GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'


@pytest.mark.ui
@pytest.mark.skipif(os.environ.get('WHYSPER_UI_MULTI_GPU_TEST') != '1', reason='Requires an isolated server with two simulated GPUs')
def test_select_gpu_and_keep_it_for_queued_job():
    upload_name = f'gpu-test-{uuid.uuid4().hex}.flac'
    with sqlite3.connect(DB_PATH) as conn:
        original = conn.execute("SELECT value FROM settings WHERE key='main'").fetchone()
    page = requests.get('http://127.0.0.1:8080/', timeout=5)
    page.raise_for_status()
    elements = json.loads(html.unescape(re.search(r'parseElements\(String.raw`(.*?)`\)', page.text, re.S).group(1)))
    client_id = re.search(r"'client_id': '([\w-]+)'", page.text).group(1)
    changed = threading.Condition()
    socket = socketio.Client(reconnection=False)
    job = None
    saved_value = None

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

    def find_label(label):
        return next(((key, item) for key, item in elements.items()
                     if item and item.get('props', {}).get('label') == label), None)

    def send(target, event_type, *args):
        key, item = target
        listener = next(event['listener_id'] for event in item['events'] if event['type'] == event_type)
        socket.emit('event', {'id': int(key), 'client_id': client_id, 'listener_id': listener,
                              'args': [json.dumps(arg) for arg in args]})

    def wait_for(predicate, message):
        deadline = time.monotonic() + 12
        while not predicate() and time.monotonic() < deadline:
            changed.wait(.1)
        assert predicate(), message

    def read_settings():
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT value FROM settings WHERE key='main'").fetchone()
        return row[0] if row else None

    def read_job():
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute('SELECT * FROM jobs WHERE name=?', (upload_name,)).fetchone()
        return dict(row) if row else None

    try:
        socket.connect('http://127.0.0.1:8080?client_id=' + client_id + '&next_message_id=0',
                       socketio_path='/_nicegui_ws/socket.io', transports=['polling'])
        assert socket.call('handshake', {'client_id': client_id, 'document_id': str(uuid.uuid4()),
                                        'tab_id': str(uuid.uuid4()), 'next_message_id': 0}, timeout=5)
        with changed:
            send(find_label('Ustawienia'), 'click')
            wait_for(lambda: find_label('Karta GPU') is not None, 'Brak wyboru GPU')
            def gpu_b_option():
                picker = find_label('Karta GPU')
                return next((item for item in picker[1]['props'].get('options', [])
                             if 'Test GPU B' in item.get('label', '')), None) if picker else None
            wait_for(lambda: gpu_b_option() is not None, 'Nie wykryto drugiej symulowanej karty')
            picker = find_label('Karta GPU')
            send(picker, 'update:modelValue', gpu_b_option())
            wait_for(lambda: 'Test GPU B' in str(find_label('Karta GPU')[1]['props'].get('model-value')),
                     'Nie wybrano drugiej karty')
            send(find_label('Zapisz ustawienia'), 'click')
            wait_for(lambda: read_settings() is not None and Settings.model_validate_json(read_settings()).gpu_id == GPU_B,
                     'Nie zapisano wybranej karty')
            saved_value = read_settings()
            send(find_label('Transkrypcja'), 'click')
            wait_for(lambda: any(item and item.get('tag') == 'nicegui-upload' for item in elements.values()),
                     'Nie otwarto formularza transkrypcji')
            wait_for(lambda: find_label('Karta GPU') is not None
                     and 'Test GPU B' in str(find_label('Karta GPU')[1]['props'].get('model-value')),
                     'Formularz nie zachował wybranej karty')
            upload = next(item for item in elements.values() if item and item.get('tag') == 'nicegui-upload')
        with (Path(__file__).parent / 'fixtures' / 'jfk.flac').open('rb') as audio:
            response = requests.post('http://127.0.0.1:8080' + upload['props']['url'],
                                     files={'file': (upload_name, audio, 'audio/flac')}, timeout=20)
        response.raise_for_status()
        with changed:
            wait_for(lambda: any(item and upload_name in item.get('text', '') for item in elements.values()),
                     'Nie dodano pliku')
            send(find_label('Dodaj do kolejki'), 'click')
            wait_for(lambda: read_job() is not None, 'Nie dodano zadania')
            job = read_job()
            assert job['status'] == 'waiting'  # The isolated test server has its workers disabled.
            assert Settings.model_validate_json(job['options']).gpu_id == GPU_B
    finally:
        if socket.connected:
            socket.disconnect()
        job = job or read_job()
        removed = False
        with sqlite3.connect(DB_PATH) as conn:
            if job:
                removed = bool(conn.execute("DELETE FROM jobs WHERE id=? AND status='waiting'", (job['id'],)).rowcount)
            current = conn.execute("SELECT value FROM settings WHERE key='main'").fetchone()
            if saved_value and current and current[0] == saved_value:
                if original:
                    conn.execute("UPDATE settings SET value=? WHERE key='main' AND value=?", (original[0], saved_value))
                else:
                    conn.execute("DELETE FROM settings WHERE key='main' AND value=?", (saved_value,))
        if removed and Path(job['path']).parent.resolve() == UPLOADS.resolve():
            Path(job['path']).unlink(missing_ok=True)


@pytest.mark.ui
@pytest.mark.skipif(os.environ.get('WHYSPER_UI_MULTI_GPU_TEST') != '1', reason='Requires an isolated server with two simulated GPUs')
def test_diagnostics_runs_test_on_selected_gpu():
    page = requests.get('http://127.0.0.1:8080/', timeout=5)
    page.raise_for_status()
    elements = json.loads(html.unescape(re.search(r'parseElements\(String.raw`(.*?)`\)', page.text, re.S).group(1)))
    client_id = re.search(r"'client_id': '([\w-]+)'", page.text).group(1)
    changed = threading.Condition()
    socket = socketio.Client(reconnection=False)

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

    def find_label(label):
        return next(((key, item) for key, item in elements.items()
                     if item and item.get('props', {}).get('label') == label), None)

    def send(target, event_type, *args):
        key, item = target
        listener = next(event['listener_id'] for event in item['events'] if event['type'] == event_type)
        socket.emit('event', {'id': int(key), 'client_id': client_id, 'listener_id': listener,
                              'args': [json.dumps(arg) for arg in args]})

    def wait_for(predicate, message):
        deadline = time.monotonic() + 12
        while not predicate() and time.monotonic() < deadline:
            changed.wait(.1)
        assert predicate(), message

    def gpu_b_option():
        picker = find_label('Karta GPU')
        return next((option for option in picker[1]['props'].get('options', [])
                     if 'Test GPU B' in option.get('label', '')), None) if picker else None

    def has_probe_result():
        for item in elements.values():
            if item and 'whitespace-pre-wrap' in item.get('class', []):
                try:
                    result = json.loads(item.get('text', ''))
                except (TypeError, ValueError):
                    continue
                if result.get('ok') and result.get('gpu_id') == GPU_B:
                    return True
        return False

    try:
        socket.connect('http://127.0.0.1:8080?client_id=' + client_id + '&next_message_id=0',
                       socketio_path='/_nicegui_ws/socket.io', transports=['polling'])
        assert socket.call('handshake', {'client_id': client_id, 'document_id': str(uuid.uuid4()),
                                        'tab_id': str(uuid.uuid4()), 'next_message_id': 0}, timeout=5)
        with changed:
            send(find_label('Diagnostyka'), 'click')
            wait_for(lambda: gpu_b_option() is not None, 'Nie wykryto drugiej symulowanej karty')
            send(find_label('Karta GPU'), 'update:modelValue', gpu_b_option())
            wait_for(lambda: 'Test GPU B' in str(find_label('Karta GPU')[1]['props'].get('model-value')),
                     'Nie wybrano drugiej karty do diagnostyki')
            send(find_label('Uruchom test CUDA'), 'click')
            wait_for(has_probe_result, 'Test CUDA nie otrzymał wybranej karty')
    finally:
        if socket.connected:
            socket.disconnect()
