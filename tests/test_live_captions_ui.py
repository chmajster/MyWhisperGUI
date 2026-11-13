"""UI test for an isolated server with a simulated microphone and model."""
import html
import json
import os
import re
import threading
import time
import uuid

import pytest
import requests
import socketio

from app.config import DATA


@pytest.mark.ui
@pytest.mark.skipif(os.environ.get('WHYSPER_UI_LIVE_TEST') != '1', reason='Requires an isolated server with a simulated microphone and model')
def test_live_caption_start_stop_and_navigation():
    page = requests.get('http://127.0.0.1:8080/', timeout=5)
    page.raise_for_status()
    elements = json.loads(html.unescape(re.search(r'parseElements\(String.raw`(.*?)`\)', page.text, re.S).group(1)))
    client_id = re.search(r"'client_id': '([\w-]+)'", page.text).group(1)
    changed = threading.Condition()
    socket = socketio.Client(reconnection=False)
    marker = DATA / 'tmp' / 'fake-mic-closed.log'
    downloads, scripts = [], []

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

    @socket.on('download')
    def download(data):
        with changed:
            downloads.append(data)
            changed.notify_all()

    @socket.on('run_javascript')
    def javascript(data):
        with changed:
            scripts.append(data.get('code', ''))
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

    def text_is(text):
        return any(item and item.get('text') == text for item in elements.values())

    def option(label, value):
        return next(item for item in find_label(label)[1]['props']['options'] if item['label'] == value)

    def closed_count():
        return int(marker.read_text(encoding='utf-8')) if marker.is_file() else 0

    try:
        socket.connect('http://127.0.0.1:8080?client_id=' + client_id + '&next_message_id=0',
                       socketio_path='/_nicegui_ws/socket.io', transports=['polling'])
        assert socket.call('handshake', {'client_id': client_id, 'document_id': str(uuid.uuid4()),
                                        'tab_id': str(uuid.uuid4()), 'next_message_id': 0}, timeout=5)
        with changed:
            send(find_label('Napisy na żywo'), 'click')
            wait_for(lambda: find_label('Mikrofon') is not None and find_label('Uruchom napisy') is not None,
                     'Nie otwarto widoku napisów')
            assert any(item['label'] == 'Testowy mikrofon' for item in find_label('Mikrofon')[1]['props']['options'])
            send(find_label('Model Whisper'), 'update:modelValue', option('Model Whisper', 'tiny'))
            send(find_label('Urządzenie'), 'update:modelValue', option('Urządzenie', 'CPU'))
            wait_for(lambda: find_label('Urządzenie')[1]['props']['model-value']['label'] == 'CPU'
                     and find_label('Model Whisper')[1]['props']['model-value']['label'] == 'tiny',
                     'Nie wybrano CPU i modelu tiny')
            send(find_label('Uruchom napisy'), 'click')
            wait_for(lambda: text_is('Testowy napis na żywo.'), 'Nie pokazano rozpoznanego tekstu')
            wait_for(lambda: not find_label('Pobierz TXT')[1]['props'].get('disable'), 'Nie włączono pobierania tekstu')
            send(find_label('Kopiuj całość'), 'click')
            wait_for(lambda: any('clipboard.writeText' in script and 'Testowy napis' in script for script in scripts),
                     'Nie wysłano tekstu do schowka')
            send(find_label('Pobierz TXT'), 'click')
            wait_for(lambda: bool(downloads), 'Nie wysłano pliku TXT')
            assert downloads[0]['filename'] == 'napisy-na-zywo.txt'
            assert downloads[0]['src'].decode('utf-8') == 'Testowy napis na żywo.\n'
            send(find_label('Zatrzymaj'), 'click')
            wait_for(lambda: text_is('Zatrzymano') and closed_count() >= 1, 'Nie zatrzymano mikrofonu')

            speaker = next((key, item) for key, item in elements.items()
                           if item and item.get('tag') == 'q-toggle' and item.get('text') == 'Rozróżniaj rozmówców')
            send(speaker, 'update:modelValue', True)
            wait_for(lambda: elements[speaker[0]]['props']['model-value'] is True, 'Nie włączono rozróżniania rozmówców')
            send(find_label('Uruchom napisy'), 'click')
            wait_for(lambda: text_is('Rozmówca 1: Testowy napis na żywo.'), 'Nie pokazano rozmówcy przy napisie')
            send(find_label('Zatrzymaj'), 'click')
            wait_for(lambda: text_is('Zatrzymano') and closed_count() >= 2, 'Nie zatrzymano kolejnej sesji')
            send(speaker, 'update:modelValue', False)
            wait_for(lambda: elements[speaker[0]]['props']['model-value'] is False, 'Nie wyłączono rozróżniania rozmówców')

            send(find_label('Uruchom napisy'), 'click')
            wait_for(lambda: text_is('Mikrofon włączony · napisy na żywo'), 'Nie uruchomiono kolejnej sesji')
            send(find_label('Transkrypcja'), 'click')
            wait_for(lambda: text_is('Nowa transkrypcja') and closed_count() >= 3,
                     'Mikrofon nie został zamknięty po zmianie zakładki')

            send(find_label('Napisy na żywo'), 'click')
            wait_for(lambda: find_label('Model Whisper') is not None and find_label('Uruchom napisy') is not None
                     and not find_label('Uruchom napisy')[1]['props'].get('disable'),
                     'Nie otwarto ponownie widoku napisów')
            send(find_label('Model Whisper'), 'update:modelValue', option('Model Whisper', 'tiny'))
            send(find_label('Urządzenie'), 'update:modelValue', option('Urządzenie', 'CPU'))
            wait_for(lambda: find_label('Urządzenie')[1]['props']['model-value']['label'] == 'CPU'
                     and find_label('Model Whisper')[1]['props']['model-value']['label'] == 'tiny'
                     and not find_label('Uruchom napisy')[1]['props'].get('disable'),
                     'Nie przygotowano kolejnej sesji')
            send(find_label('Uruchom napisy'), 'click')
            wait_for(lambda: text_is('Mikrofon włączony · napisy na żywo'), 'Nie uruchomiono kolejnej sesji')
        socket.disconnect()
        deadline = time.monotonic() + 12
        while closed_count() < 4 and time.monotonic() < deadline:
            time.sleep(.1)
        assert closed_count() >= 4, 'Mikrofon nie został zamknięty po rozłączeniu przeglądarki'
    finally:
        if socket.connected:
            socket.disconnect()
