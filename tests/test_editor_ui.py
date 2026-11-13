"""Exercise long transcripts through the same UI events as a browser."""
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

from app.config import DB_PATH, Settings
from app.ui.layout import EDITOR_PAGE_SIZE


@pytest.mark.ui
@pytest.mark.skipif(os.environ.get('WHYSPER_UI_TEST') != '1', reason='Set WHYSPER_UI_TEST=1 with the app running')
@pytest.mark.parametrize('media_present', [True, False])
def test_long_transcript_pagination_search_and_save(media_present):
    job_id = uuid.uuid4().hex
    name = f'editor-test-{job_id}.flac'
    source = Path(__file__).parent / 'fixtures' / ('jfk.flac' if media_present else f'missing-{job_id}.flac')
    segments = [{'start': float(index), 'end': index + .5, 'text': f'Fragment {index:04}'} for index in range(2500)]
    segments[10]['text'] = 'needle-0010'
    segments[-1]['text'] = 'needle-2499'
    result = {'model': 'tiny', 'device': 'cpu', 'language': 'pl', 'segments': segments}
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''INSERT INTO jobs(id,name,path,status,position,created_at,updated_at,options,result)
                        VALUES (?,?,?,?,?,?,?,?,?)''',
                     (job_id, name, str(source), 'done', 999999, '2026', '2026',
                      Settings().model_dump_json(), json.dumps(result)))

    socket = socketio.Client(reconnection=False)
    changed = threading.Condition()
    elements = {}

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

    def emit_event(element_id, element, event_type, *args):
        listener = next(event['listener_id'] for event in element['events'] if event['type'] == event_type)
        socket.emit('event', {'id': int(element_id), 'client_id': client_id, 'listener_id': listener,
                              'args': [json.dumps(arg) for arg in args]})

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

    def fields():
        return [(key, item) for key, item in elements.items()
                if item and item.get('tag') == 'nicegui-input' and item.get('props', {}).get('type') == 'textarea']

    def contains(text):
        return any(item and item.get('text') == text for item in elements.values())

    def saved_text(index):
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute('SELECT result FROM jobs WHERE id=?', (job_id,)).fetchone()
        return json.loads(row[0])['segments'][index]['text']

    def saved_speaker(index):
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute('SELECT result FROM jobs WHERE id=?', (job_id,)).fetchone()
        return json.loads(row[0])['segments'][index].get('speaker')

    try:
        page = requests.get('http://127.0.0.1:8080/', timeout=5)
        page.raise_for_status()
        elements.update(json.loads(html.unescape(re.search(r'parseElements\(String.raw`(.*?)`\)', page.text, re.S).group(1))))
        client_id = re.search(r"'client_id': '([\w-]+)'", page.text).group(1)
        socket.connect(f'http://127.0.0.1:8080?client_id={client_id}&next_message_id=0',
                       socketio_path='/_nicegui_ws/socket.io', transports=['polling'])
        assert socket.call('handshake', {'client_id': client_id, 'document_id': str(uuid.uuid4()),
                                        'tab_id': str(uuid.uuid4()), 'next_message_id': 0}, timeout=5)
        with changed:
            nav = next((key, item) for key, item in elements.items()
                       if item and item.get('props', {}).get('label') == 'Historia')
            emit_event(*nav, 'click')
            wait_for(lambda: contains(name), 'Nie wyświetlono zadania w historii')
            row = next(key for key, item in elements.items() if item and 'job-row' in item.get('class', [])
                       and any(child.get('text') == name for child in descendants(key)))
            button = next((key, item) for key, item in elements.items() if item in descendants(row)
                          and item.get('props', {}).get('label') == 'Otwórz')
            emit_event(*button, 'click')
            wait_for(lambda: contains(f'Fragmenty 1–{EDITOR_PAGE_SIZE} z 2500'), 'Edytor nie otworzył pierwszej strony')
            wait_for(lambda: len(fields()) == EDITOR_PAGE_SIZE, 'Nieprawidłowa liczba pól na stronie')
            assert len(fields()) == EDITOR_PAGE_SIZE
            assert all('debounce' not in item['props'] for _, item in fields())
            assert any(item.get('props', {}).get('value') == 'Fragment 0000' for _, item in fields())
            assert not any(item.get('props', {}).get('value') == 'Fragment 0050' for _, item in fields())
            has_player = any(item and item.get('tag') == 'nicegui-audio' for item in elements.values())
            assert has_player == media_present
            if not media_present:
                assert contains('Nie znaleziono pliku nagrania w zapisanej lokalizacji. Transkrypcję można nadal edytować i eksportować.')

            pager = next((key, item) for key, item in elements.items() if item and item.get('tag') == 'q-pagination')
            assert pager[1]['props']['max'] == 50
            emit_event(*pager, 'update:modelValue', 2)
            wait_for(lambda: contains('Fragmenty 51–100 z 2500'), 'Nie zmieniono strony')
            wait_for(lambda: any(item.get('props', {}).get('value') == 'Fragment 0050' for _, item in fields()),
                     'Brak fragmentu z drugiej strony')
            field = next((key, item) for key, item in fields() if item.get('props', {}).get('value') == 'Fragment 0050')
            emit_event(*field, 'update:value', 'Poprawiony fragment 0050')
            pager = next((key, item) for key, item in elements.items() if item and item.get('tag') == 'q-pagination')
            emit_event(*pager, 'update:modelValue', 1)
            wait_for(lambda: contains('Fragmenty 1–50 z 2500'), 'Nie wrócono do pierwszej strony')
            wait_for(lambda: saved_text(50) == 'Poprawiony fragment 0050', 'Zmiana nie została zapisana przed przejściem dalej')

            search = next((key, item) for key, item in elements.items()
                          if item and item.get('props', {}).get('placeholder') == 'Szukaj w transkrypcji…')
            emit_event(*search, 'update:value', 'needle')
            wait_for(lambda: contains('Fragmenty 1–2 z 2'), 'Wyszukiwanie nie objęło całego wyniku')
            wait_for(lambda: len(fields()) == 2, 'Nieprawidłowa liczba znalezionych fragmentów')
            field = next((key, item) for key, item in fields() if item.get('props', {}).get('value') == 'needle-2499')
            emit_event(*field, 'update:value', 'needle-poprawiony')
            wait_for(lambda: saved_text(2499) == 'needle-poprawiony', 'Edycja wyniku wyszukiwania nie została zapisana')
            assert saved_text(10) == 'needle-0010'
            emit_event(*search, 'update:value', None)
            wait_for(lambda: contains('Fragmenty 1–50 z 2500'), 'Wyczyszczenie wyszukiwania nie przywróciło stron')
            toggle = next((key, item) for key, item in elements.items()
                          if item and item.get('tag') == 'q-toggle' and item.get('text') == 'Edytuj rozmówców')
            emit_event(*toggle, 'update:modelValue', True)
            def speaker_fields():
                return [(key, item) for key, item in elements.items()
                        if item and item.get('props', {}).get('placeholder') == 'Nieustalony']
            wait_for(lambda: len(speaker_fields()) == EDITOR_PAGE_SIZE, 'Nie pokazano pól rozmówców')
            row = next(key for key, item in elements.items() if item and 'job-row' in item.get('class', [])
                       and any(child.get('props', {}).get('value') == 'Fragment 0000' for child in descendants(key)))
            field = next((key, item) for key, item in speaker_fields() if item in descendants(row))
            emit_event(*field, 'update:value', 'Anna')
            wait_for(lambda: saved_speaker(0) == 'Anna', 'Nie zapisano nazwy rozmówcy')
            emit_event(*search, 'update:value', 'Anna')
            wait_for(lambda: contains('Fragmenty 1–1 z 1'), 'Nie znaleziono fragmentów rozmówcy')

        response = requests.get(f'http://127.0.0.1:8080/export/{job_id}/txt', timeout=5)
        response.raise_for_status()
        lines = response.text.splitlines()
        assert len(lines) == 2500
        assert lines[0] == 'Anna: Fragment 0000'
        assert lines[50] == 'Poprawiony fragment 0050'
        assert lines[2499] == 'needle-poprawiony'
    finally:
        if socket.connected:
            socket.disconnect()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('DELETE FROM jobs WHERE id=?', (job_id,))
