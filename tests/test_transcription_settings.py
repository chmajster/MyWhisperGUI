import inspect
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions
from nicegui import ui
from pydantic import ValidationError

from app.config import Settings
from app.core import audio_processor, whisper_engine
from app.core.whisper_engine import WhisperEngine
from app.database.db import Database
from app.ui.layout import form_values, option_form


def test_existing_settings_keep_faster_whisper_defaults():
    settings = Settings.model_validate_json('{"model":"tiny","beam_size":3,"temperature":0,"vad":true}')
    defaults = inspect.signature(WhisperModel.transcribe).parameters
    for name in ('best_of', 'patience', 'repetition_penalty', 'no_repeat_ngram_size',
                 'no_speech_threshold', 'condition_on_previous_text'):
        assert getattr(settings, name) == defaults[name].default
    vad = VadOptions()
    assert settings.vad_threshold == vad.threshold
    assert settings.vad_min_silence_ms == vad.min_silence_duration_ms
    assert settings.vad_speech_pad_ms == vad.speech_pad_ms
    assert settings.hotwords == ''
    assert settings.beam_size == 3


def test_cpu_requests_the_concrete_int8_precision(monkeypatch, tmp_path):
    calls = []

    class FakeModel:
        def __init__(self, path, **kwargs):
            calls.append(kwargs)
            self.model = SimpleNamespace(device='cpu', compute_type='int8_float32', unload_model=lambda **_: None)

    monkeypatch.setattr(whisper_engine, 'configure_cuda_dlls', lambda: None)
    monkeypatch.setitem(sys.modules, 'ctranslate2', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=FakeModel))
    engine = WhisperEngine(SimpleNamespace(path=lambda *_: tmp_path))
    model, precision = engine._model(Settings(model='tiny', device='cpu'))
    assert model.model.compute_type == precision == 'int8_float32'
    assert calls[0]['compute_type'] == 'int8_float32'
    engine.unload()


@pytest.mark.parametrize('include_paths', [False, True])
def test_form_controls_legend_and_settings_persistence(tmp_path, include_paths):
    original = Settings(device='cpu', model='tiny')
    with ui.card() as panel:
        fields = option_form(original, include_paths=include_paths)
    try:
        assert fields['beam_size'].enabled and fields['patience'].enabled
        assert not fields['best_of'].enabled
        assert all(fields[name].enabled for name in ('vad_threshold', 'vad_min_silence_ms', 'vad_speech_pad_ms'))
        legend = next(item for item in panel.descendants()
                      if item.tag == 'q-expansion-item' and item._props.get('label') == 'Legenda parametrów')
        for title in ('Beam size', 'Temperature', 'Best of', 'Filtr ciszy VAD i próg mowy', 'Kontekst początkowy i słowa podpowiedzi'):
            assert any(getattr(item, 'text', None) == title for item in legend.descendants())

        fields['temperature'].value = .4
        assert not fields['beam_size'].enabled and not fields['patience'].enabled
        assert fields['best_of'].enabled
        fields['best_of'].value = 3
        fields['patience'].value = 1.5
        fields['repetition_penalty'].value = 1.4
        fields['no_repeat_ngram_size'].value = 4
        fields['no_speech_threshold'].value = .7
        fields['condition_on_previous_text'].value = False
        fields['vad_threshold'].value = .3
        fields['vad_min_silence_ms'].value = 700
        fields['vad_speech_pad_ms'].value = 250
        fields['hotwords'].value = 'Kowalski, Whisper'
        fields['vad'].value = False
        assert all(not fields[name].enabled for name in ('vad_threshold', 'vad_min_silence_ms', 'vad_speech_pad_ms'))
        updated = form_values(fields, original)
        assert updated.vad is False and updated.vad_min_silence_ms == 700
        assert updated.best_of == 3 and updated.hotwords == 'Kowalski, Whisper'

        fields['vad'].value = True
        assert fields['vad_threshold'].enabled and fields['vad_threshold'].value == .3
        fields['temperature'].value = 0
        assert fields['beam_size'].enabled and fields['patience'].enabled and not fields['best_of'].enabled

        database = Database(tmp_path / 'settings.sqlite')
        database.save_settings(updated)
        assert database.settings() == updated
        job = database.add_job('test.wav', 'test.wav', updated)
        assert Settings.model_validate_json(database.get(job)['options']) == updated
    finally:
        panel.delete()


@pytest.mark.parametrize('field,value', [
    ('best_of', 0), ('best_of', 11), ('patience', 0), ('patience', 5.1),
    ('repetition_penalty', .9), ('repetition_penalty', 3.1),
    ('no_repeat_ngram_size', -1), ('no_repeat_ngram_size', 11),
    ('no_speech_threshold', -0.1), ('no_speech_threshold', 1.1),
    ('vad_threshold', 0), ('vad_threshold', 1),
    ('vad_min_silence_ms', 99), ('vad_min_silence_ms', 10001),
    ('vad_speech_pad_ms', -1), ('vad_speech_pad_ms', 1501),
    ('hotwords', 'x' * 4001),
])
def test_rejects_invalid_advanced_settings(field, value):
    with pytest.raises(ValidationError):
        Settings.model_validate({field: value})


@pytest.mark.parametrize('vad_enabled', [True, False])
def test_engine_passes_settings_to_each_audio_chunk(monkeypatch, vad_enabled):
    calls = []

    class FakeModel:
        def transcribe(self, samples, **kwargs):
            calls.append(kwargs)
            segment = SimpleNamespace(start=0.0, end=.5, text=' test ', words=None)
            return iter([segment]), SimpleNamespace(language='pl', language_probability=.95)

    model = FakeModel()
    monkeypatch.setattr(WhisperEngine, '_model', lambda self, options: (model, 'int8'))
    monkeypatch.setattr(audio_processor, 'chunks', lambda path: iter([(0, np.zeros(8000)), (.5, np.zeros(8000))]))
    options = Settings(model='tiny', device='cpu', keep_model=True, beam_size=7, temperature=.4,
                       best_of=3, patience=1.5, repetition_penalty=1.4, no_repeat_ngram_size=4,
                       no_speech_threshold=.7, condition_on_previous_text=False, vad=vad_enabled,
                       vad_threshold=.3, vad_min_silence_ms=700, vad_speech_pad_ms=250,
                       initial_prompt='Temat nagrania', hotwords='Kowalski, Whisper')
    result = WhisperEngine(None).transcribe('not-read.wav', options, lambda value: None, lambda: False)
    assert len(result['segments']) == len(calls) == 2
    assert calls[0]['language'] is None and calls[1]['language'] == 'pl'
    for call in calls:
        for name in ('beam_size', 'temperature', 'best_of', 'patience', 'repetition_penalty',
                     'no_repeat_ngram_size', 'no_speech_threshold', 'condition_on_previous_text',
                     'initial_prompt', 'hotwords'):
            assert call[name] == getattr(options, name)
        assert call['vad_filter'] is vad_enabled
        expected_vad = {'threshold': .3, 'min_silence_duration_ms': 700, 'speech_pad_ms': 250} if vad_enabled else None
        assert call['vad_parameters'] == expected_vad
