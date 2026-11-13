"""Exercise server-side UI events without requiring a graphical desktop."""
import html
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import pytest
import requests
import socketio

from app.config import DB_PATH, MODEL_REPOS, UPLOADS, Settings
from app.core.model_manager import ModelManager


@pytest.mark.ui
@pytest.mark.skipif(os.environ.get('WHYSPER_UI_TEST') != '1', reason='Set WHYSPER_UI_TEST=1 with the app running')
def test_ui_navigation_over_localhost():
    page = requests.get('http://127.0.0.1:8080/', timeout=5)
    page.raise_for_status()
    match = re.search(r'parseElements\(String.raw`(.*?)`\)', page.text, re.S)
    assert match, 'NiceGUI element tree was not rendered'
    elements = json.loads(html.unescape(match.group(1)))
    client_id = re.search(r"'client_id': '([\w-]+)'", page.text).group(1)
    full = os.environ.get('WHYSPER_UI_FULL_TEST') == '1'
    upload_name = f'ui-test-{uuid.uuid4().hex}.flac'
    test_job = None
    changed = threading.Condition()
    socket = socketio.Client(reconnection=False)
    scripts = []

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

    @socket.on('run_javascript')
    def javascript(data):
        with changed:
            scripts.append(data.get('code', ''))
            changed.notify_all()

    def children(root):
        result = []
        todo = [root]
        while todo:
            item = elements.get(str(todo.pop()))
            if item:
                result.append(item)
                todo.extend(item.get('children', []))
        return result

    socket.connect(f'http://127.0.0.1:8080?client_id={client_id}&next_message_id=0',
                   socketio_path='/_nicegui_ws/socket.io', transports=['polling'])
    try:
        assert socket.call('handshake', {'client_id': client_id, 'document_id': str(uuid.uuid4()),
                                        'tab_id': str(uuid.uuid4()), 'next_message_id': 0}, timeout=5)
        for label, expected in [('Transkrypcja', 'Nowa transkrypcja'), ('Nagrywanie', 'Nagrywanie'),
                                ('Kolejka', 'Kolejka zadań'), ('Historia', 'Historia'),
                                ('Modele AI', 'Modele AI'), ('Ustawienia', 'Ustawienia'),
                                ('Diagnostyka', 'Diagnostyka'), ('Dashboard', 'Twoje centrum transkrypcji')]:
            with changed:
                buttons = [(int(key), item) for key, item in elements.items()
                           if isinstance(item, dict) and item.get('props', {}).get('label') == label
                           and 'nav-btn' in item.get('class', [])]
                assert len(buttons) == 1, label
                element_id, button = buttons[0]
                listener = next(event['listener_id'] for event in button['events'] if event['type'] == 'click')
                socket.emit('event', {'id': element_id, 'client_id': client_id, 'listener_id': listener, 'args': []})
                deadline = time.monotonic() + 5
                while not any(item and item.get('text') == expected and 'page-title' in item.get('class', [])
                              for item in elements.values()) and time.monotonic() < deadline:
                    changed.wait(.1)
                assert any(item and item.get('text') == expected and 'page-title' in item.get('class', [])
                           for item in elements.values()), f'Page failed to render: {label}'
            if label == 'Transkrypcja':
                upload = next(item for item in elements.values() if item and item.get('tag') == 'nicegui-upload')
                with (Path(__file__).parent / 'fixtures' / 'jfk.flac').open('rb') as audio:
                    response = requests.post('http://127.0.0.1:8080' + upload['props']['url'],
                                             files={'file': (upload_name, audio, 'audio/flac')}, timeout=20)
                response.raise_for_status()
                with changed:
                    deadline = time.monotonic() + 15
                    while not any(upload_name in item.get('text', '') for item in elements.values() if item) and time.monotonic() < deadline:
                        changed.wait(.1)
                    assert any(upload_name in item.get('text', '') for item in elements.values() if item), 'Uploaded file was not processed'
                if full:
                    with changed:
                        model_id, model = next((int(key), item) for key, item in elements.items()
                                               if item and item.get('props', {}).get('label') == 'Model Whisper')
                        listener = next(e['listener_id'] for e in model['events'] if e['type'] == 'update:modelValue')
                        socket.emit('event', {'id': model_id, 'client_id': client_id, 'listener_id': listener,
                                              'args': [json.dumps({'value': 0, 'label': 'tiny'})]})
                        deadline = time.monotonic() + 5
                        while elements[str(model_id)]['props']['model-value']['label'] != 'tiny' and time.monotonic() < deadline:
                            changed.wait(.1)
                        assert elements[str(model_id)]['props']['model-value']['label'] == 'tiny'
                        button_id, button = next((int(k), v) for k, v in elements.items() if v and v.get('props', {}).get('label') == 'Dodaj do kolejki')
                        socket.emit('event', {'id': button_id, 'client_id': client_id,
                                              'listener_id': button['events'][0]['listener_id'], 'args': []})
                        deadline = time.monotonic() + 5
                        while not any(v and v.get('text') == 'Kolejka zadań' and 'page-title' in v.get('class', []) for v in elements.values()) and time.monotonic() < deadline:
                            changed.wait(.1)
                    deadline = time.monotonic() + 180
                    while time.monotonic() < deadline:
                        with sqlite3.connect(DB_PATH) as conn:
                            conn.row_factory = sqlite3.Row
                            found = conn.execute('SELECT * FROM jobs WHERE name=?', (upload_name,)).fetchone()
                        if found:
                            test_job = dict(found)
                            if test_job['status'] in ('done', 'error'):
                                break
                        time.sleep(.1)
                    assert test_job and test_job['status'] == 'done', test_job
                    result = json.loads(test_job['result'])
                    assert result['device'] == 'cuda' and result['segments']
                    response = requests.get(f"http://127.0.0.1:8080/export/{test_job['id']}/txt", timeout=5)
                    response.raise_for_status()
                    assert 'country' in response.content.decode('utf-8').lower()
            if label == 'Historia' and full:
                with changed:
                    row = next(key for key, item in elements.items() if item and 'job-row' in item.get('class', [])
                               and any(child.get('text') == upload_name for child in children(key)))
                    target = next((int(key), item) for key, item in elements.items() if item in children(row)
                                  and item.get('props', {}).get('label') == 'Otwórz')
                    socket.emit('event', {'id': target[0], 'client_id': client_id,
                                          'listener_id': target[1]['events'][0]['listener_id'], 'args': []})
                    deadline = time.monotonic() + 5
                    while not any(item and item.get('text') == 'Edytor transkrypcji' for item in elements.values()) and time.monotonic() < deadline:
                        changed.wait(.1)
                    assert any(item and item.get('text') == 'Edytor transkrypcji' for item in elements.values())
                    deadline = time.monotonic() + 5
                    matches = []
                    while not matches and time.monotonic() < deadline:
                        matches = [(int(key), item) for key, item in elements.items() if item
                                   and 'country' in str(item.get('props', {}).get('value', '')).lower()]
                        if not matches:
                            changed.wait(.1)
                    assert matches, 'Editor text was not rendered'
                    text_id, text = matches[0]
                    listener = next(e['listener_id'] for e in text['events'] if e['type'] == 'update:value')
                    socket.emit('event', {'id': text_id, 'client_id': client_id, 'listener_id': listener,
                                          'args': [json.dumps('Poprawiona transkrypcja — zażółć gęślą jaźń.')]})
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    response = requests.get(f"http://127.0.0.1:8080/export/{test_job['id']}/txt", timeout=5)
                    if 'zażółć' in response.content.decode('utf-8').lower():
                        break
                    time.sleep(.1)
                assert 'zażółć' in response.content.decode('utf-8').lower()
                with changed:
                    copy_id, copy = next((int(k), v) for k, v in elements.items() if v and v.get('props', {}).get('label') == 'Kopiuj całość')
                    socket.emit('event', {'id': copy_id, 'client_id': client_id,
                                          'listener_id': copy['events'][0]['listener_id'], 'args': []})
                    deadline = time.monotonic() + 5
                    while not any('clipboard.writeText' in script for script in scripts) and time.monotonic() < deadline:
                        changed.wait(.1)
                    assert any('clipboard.writeText' in script for script in scripts)
            if label == 'Modele AI':
                with sqlite3.connect(DB_PATH) as conn:
                    saved = conn.execute("SELECT value FROM settings WHERE key='main'").fetchone()
                settings = Settings.model_validate_json(saved[0]) if saved else Settings()
                manager = ModelManager()
                available = [name for name in MODEL_REPOS if manager.available(name, settings.cache_dir)]
                with changed:
                    deadline = time.monotonic() + 5
                    summary = f'Pobrane: {len(available)} z {len(MODEL_REPOS)} modeli'
                    while not any(item and item.get('text') == summary for item in elements.values()) and time.monotonic() < deadline:
                        changed.wait(.1)
                    assert any(item and item.get('text') == summary for item in elements.values())
                    for model in available:
                        row = next(key for key, item in elements.items() if item and 'model-row' in item.get('class', [])
                                   and any(child.get('text') == model for child in children(key)))
                        assert any(child.get('text') == 'POBRANY · GOTOWY OFFLINE' for child in children(row))
                        assert any(child.get('props', {}).get('label') == 'Pobrano' and child.get('props', {}).get('disable')
                                   for child in children(row))
    finally:
        socket.disconnect()
        if test_job and test_job['status'] in ('done', 'error'):
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('DELETE FROM job_segments WHERE job_id=?', (test_job['id'],))
                conn.execute('DELETE FROM jobs WHERE id=?', (test_job['id'],))
            source = Path(test_job['path']).resolve()
            if source.parent == UPLOADS.resolve():
                source.unlink(missing_ok=True)
            output = Path(Settings.model_validate_json(test_job['options']).output_dir)
            for extension in ('txt', 'srt', 'vtt', 'json', 'csv'):
                (output / f"{test_job['id']}.{extension}").unlink(missing_ok=True)


@pytest.mark.ui
@pytest.mark.skipif(os.environ.get('WHYSPER_UI_TEST') != '1', reason='Set WHYSPER_UI_TEST=1 with the app running')
def test_live_view_updates_and_displays_eta():
    job_id = uuid.uuid4().hex
    name = f'live-test-{job_id}.wav'
    segment = {'start': 0, 'end': 12, 'text': 'Pierwszy rozpoznany fragment.'}
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''INSERT INTO jobs(id,name,path,status,position,created_at,updated_at,options,progress,duration,eta_seconds)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                     (job_id, name, 'unused', 'transcribing', 999999, '2026', '2026', Settings().model_dump_json(), .25, 120, 90))
        conn.execute('INSERT INTO job_segments VALUES (?,?,?)', (job_id, 0, json.dumps(segment, ensure_ascii=False)))
    socket = socketio.Client(reconnection=False)
    try:
        page = requests.get('http://127.0.0.1:8080/', timeout=5)
        page.raise_for_status()
        elements = json.loads(html.unescape(re.search(r'parseElements\(String.raw`(.*?)`\)', page.text, re.S).group(1)))
        client_id = re.search(r"'client_id': '([\w-]+)'", page.text).group(1)
        changed = threading.Condition()

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

        def contains(text):
            return any(item and text in item.get('text', '') for item in elements.values())

        def wait_for(predicate, message):
            deadline = time.monotonic() + 8
            while not predicate() and time.monotonic() < deadline:
                changed.wait(.1)
            assert predicate(), message

        socket.connect(f'http://127.0.0.1:8080?client_id={client_id}&next_message_id=0',
                       socketio_path='/_nicegui_ws/socket.io', transports=['polling'])
        assert socket.call('handshake', {'client_id': client_id, 'document_id': str(uuid.uuid4()),
                                        'tab_id': str(uuid.uuid4()), 'next_message_id': 0}, timeout=5)
        with changed:
            queue = next((key, item) for key, item in elements.items() if item and item.get('props', {}).get('label') == 'Kolejka')
            click(*queue)
            wait_for(lambda: contains(name), 'Job did not appear in the queue')
            def descendants(root):
                result = []
                todo = [root]
                while todo:
                    item = elements.get(str(todo.pop()))
                    if item:
                        result.append(item)
                        todo.extend(item.get('children', []))
                return result
            row = next(key for key, item in elements.items() if item and 'job-row' in item.get('class', [])
                       and any(child.get('text') == name for child in descendants(key)))
            buttons = [(key, item) for key, item in elements.items() if item in descendants(row)
                       and item.get('props', {}).get('label') == 'Podgląd na żywo']
            assert buttons, 'Live view button was not displayed'
            click(*buttons[0])
            wait_for(lambda: contains('Pierwszy rozpoznany fragment.'), 'Initial segment was not rendered')
            wait_for(lambda: contains('ETA: około 00:01:30'), 'Estimated time was not rendered')
        second = {'start': 13, 'end': 20, 'text': 'Następny fragment na żywo.'}
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('INSERT INTO job_segments VALUES (?,?,?)', (job_id, 1, json.dumps(second, ensure_ascii=False)))
            conn.execute('UPDATE jobs SET progress=?,eta_seconds=? WHERE id=?', (.5, 45, job_id))
        with changed:
            wait_for(lambda: contains('Następny fragment na żywo.'), 'New segment did not appear')
            wait_for(lambda: contains('ETA: około 00:00:45'), 'Updated ETA did not appear')
    finally:
        if socket.connected:
            socket.disconnect()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('DELETE FROM job_segments WHERE job_id=?', (job_id,))
            conn.execute('DELETE FROM jobs WHERE id=?', (job_id,))
