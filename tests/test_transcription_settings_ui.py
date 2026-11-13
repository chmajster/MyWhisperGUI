"""Check the parameter form using NiceGUI's browser event protocol."""
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


@pytest.mark.ui
@pytest.mark.skipif(os.environ.get('WHYSPER_UI_TEST') != '1', reason='Set WHYSPER_UI_TEST=1 with the app running')
def test_transcription_options_and_legend_over_localhost():
    response = requests.get('http://127.0.0.1:8080/', timeout=5)
    response.raise_for_status()
    elements = json.loads(html.unescape(re.search(r'parseElements\(String.raw`(.*?)`\)', response.text, re.S).group(1)))
    client_id = re.search(r"'client_id': '([\w-]+)'", response.text).group(1)
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
        deadline = time.monotonic() + 8
        while not predicate() and time.monotonic() < deadline:
            changed.wait(.1)
        assert predicate(), message

    try:
        socket.connect('http://127.0.0.1:8080?client_id=' + client_id + '&next_message_id=0',
                       socketio_path='/_nicegui_ws/socket.io', transports=['polling'])
        assert socket.call('handshake', {'client_id': client_id, 'document_id': str(uuid.uuid4()),
                                        'tab_id': str(uuid.uuid4()), 'next_message_id': 0}, timeout=5)
        with changed:
            send(find_label('Transkrypcja'), 'click')
            wait_for(lambda: find_label('Legenda parametrów') is not None, 'Nie wyświetlono legendy')
            advanced = find_label('Dodatkowe parametry transkrypcji')
            legend = find_label('Legenda parametrów')
            send(advanced, 'update:modelValue', True)
            send(legend, 'update:modelValue', True)
            wait_for(lambda: elements[advanced[0]]['props']['model-value'] is True
                     and elements[legend[0]]['props']['model-value'] is True, 'Nie otwarto sekcji')
            assert any(item and item.get('text') == 'Beam size' for item in elements.values())
            assert find_label('Best of') and find_label('Próg mowy VAD') and find_label('Słowa podpowiedzi (opcjonalnie)')
            speaker = next((key, item) for key, item in elements.items()
                           if item and item.get('tag') == 'q-toggle' and item.get('text') == 'Rozróżniaj rozmówców')
            send(speaker, 'update:modelValue', True)
            wait_for(lambda: elements[speaker[0]]['props']['model-value'] is True, 'Nie włączono rozróżniania rozmówców')

            device = find_label('Urządzenie')
            original_device = device[1]['props']['model-value']
            vulkan = next(option for option in device[1]['props']['options'] if option['label'] == 'GPU · Vulkan')
            send(device, 'update:modelValue', vulkan)
            wait_for(lambda: not find_label('Numer GPU Vulkan')[1]['props'].get('disable')
                     and find_label('Kara powtórzeń')[1]['props'].get('disable'), 'Nie przełączono ustawień Vulkan')
            send(find_label('Urządzenie'), 'update:modelValue', original_device)
            wait_for(lambda: find_label('Numer GPU Vulkan')[1]['props'].get('disable'), 'Nie wyłączono pola Vulkan')

            temperature = find_label('Temperature')
            send(temperature, 'update:modelValue', .4)
            wait_for(lambda: bool(find_label('Beam size')[1]['props'].get('disable'))
                     and not find_label('Best of')[1]['props'].get('disable'), 'Pola nie dostosowały się do temperatury')
            vad = next((key, item) for key, item in elements.items()
                       if item and item.get('tag') == 'q-toggle' and item.get('text') == 'Filtr ciszy VAD')
            send(vad, 'update:modelValue', False)
            wait_for(lambda: bool(find_label('Próg mowy VAD')[1]['props'].get('disable')),
                     'Pola VAD nie wyłączyły się po wyłączeniu filtra')

            send(find_label('Ustawienia'), 'click')
            wait_for(lambda: find_label('Legenda parametrów') is not None
                     and find_label('Katalog modeli') is not None, 'Brak parametrów w ustawieniach domyślnych')
            assert find_label('Best of') and find_label('Minimalna cisza VAD (ms)')
    finally:
        if socket.connected:
            socket.disconnect()
