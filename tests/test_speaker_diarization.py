import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from app.core import speaker_diarization as speakers
from app.core import task_queue
from app.core.exporter import render
from app.core.audio_processor import chunks
from app.core.whisper_engine import WhisperEngine
from app.database.db import Database


def test_word_alignment_splits_turns_preserves_text_and_marks_uncertainty():
    segments = [{'start': 10.0, 'end': 14.0, 'text': 'Cześć! Dzień dobry. Nie wiem.', 'words': [
        {'start': 10.1, 'end': 10.8, 'word': ' Cześć!'},
        {'start': 11.0, 'end': 11.5, 'word': ' Dzień'},
        {'start': 11.5, 'end': 12.1, 'word': ' dobry.'},
        {'start': 13.5, 'end': 13.9, 'word': ' Nie wiem.'},
    ]}]
    turns = [{'start': 0, 'end': .9, 'speaker': 'Rozmówca 1'},
             {'start': .9, 'end': 2.2, 'speaker': 'Rozmówca 2'},
             {'start': 3.4, 'end': 4, 'speaker': None}]
    result = speakers.label_segments(segments, turns, offset=10)
    assert [(s['speaker'], s['text']) for s in result] == [
        ('Rozmówca 1', 'Cześć!'), ('Rozmówca 2', 'Dzień dobry.'), (None, 'Nie wiem.')]
    assert [(s['start'], s['end']) for s in result] == [(10.1, 10.8), (11.0, 12.1), (13.5, 13.9)]
    assert 'words' not in result[0]
    assert len(speakers.label_segments(segments, turns, offset=10, keep_words=True)[1]['words']) == 2


def test_overlapping_speakers_are_not_guessed():
    turns = [{'start': 0, 'end': 1, 'speaker': 'Rozmówca 1'},
             {'start': 0, 'end': 1, 'speaker': 'Rozmówca 2'}]
    assert speakers.label_segments([{'start': .2, 'end': .8, 'text': 'Jednocześnie'}], turns)[0]['speaker'] is None


def test_incomplete_word_timings_do_not_remove_transcribed_text():
    segment = {'start': 0, 'end': 2, 'text': 'Pełna wypowiedź',
               'words': [{'start': 0, 'end': 1, 'word': ' Pełna'}]}
    turns = [{'start': 0, 'end': 2, 'speaker': 'Rozmówca 1'}]
    result = speakers.label_segments([segment], turns)
    assert result == [{'start': 0, 'end': 2, 'text': 'Pełna wypowiedź', 'speaker': 'Rozmówca 1'}]


def test_speaker_ids_remain_stable_and_distinct_between_chunks():
    tracker = speakers.SpeakerDiarizer.__new__(speakers.SpeakerDiarizer)
    tracker.profiles = []
    first = tracker._match({0: np.array([1., 0.]), 1: np.array([0., 1.])})
    second = tracker._match({7: np.array([0., 1.]), 8: np.array([1., 0.])})
    assert first == {0: 'Rozmówca 1', 1: 'Rozmówca 2'}
    assert second == {7: 'Rozmówca 2', 8: 'Rozmówca 1'}
    assert tracker._match({2: None}) == {}


def test_model_download_is_verified_cached_and_cancellable(tmp_path, monkeypatch):
    body = b'test model data'
    digest = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(speakers, '_MODELS', (('segmentation.onnx', 'https://example.test/a', digest),
                                             ('embedding.onnx', 'https://example.test/b', digest)))
    calls = []
    def fake_open(url, timeout):
        calls.append(url)
        return io.BytesIO(body)
    monkeypatch.setattr(speakers, 'urlopen', fake_open)
    first, second = speakers.prepare_models(tmp_path)
    assert first.read_bytes() == second.read_bytes() == body
    speakers.prepare_models(tmp_path)
    assert len(calls) == 2
    first.write_bytes(b'corrupt')
    monkeypatch.setattr(speakers, 'urlopen', lambda *args, **kwargs: io.BytesIO(b'incorrect'))
    with pytest.raises(RuntimeError, match='integralności'):
        speakers.prepare_models(tmp_path)
    assert first.read_bytes() == b'corrupt'
    assert not list(first.parent.glob('*.part'))
    with pytest.raises(speakers.SpeakerCancelled):
        speakers.prepare_models(tmp_path, lambda: True)


def test_whisper_stream_splits_and_publishes_speaker_segments(monkeypatch):
    class FakeModel:
        def transcribe(self, samples, **kwargs):
            assert kwargs['word_timestamps'] is True
            words = [SimpleNamespace(start=.1, end=.8, word=' Pierwszy', probability=.9),
                     SimpleNamespace(start=1.1, end=1.8, word=' Drugi', probability=.9)]
            segment = SimpleNamespace(start=.1, end=1.8, text='Pierwszy Drugi', words=words)
            return iter([segment]), SimpleNamespace(language='pl', language_probability=.99)

    class FakeTracker:
        def turns(self, samples, cancelled):
            return [{'start': 0, 'end': 1, 'speaker': 'Rozmówca 1'},
                    {'start': 1, 'end': 2, 'speaker': 'Rozmówca 2'}]

    engine = WhisperEngine(None)
    monkeypatch.setattr(engine, '_model', lambda _: (FakeModel(), 'int8'))
    emitted = []
    result = engine._transcribe_chunks([(30., np.zeros(32000, dtype=np.float32))],
        Settings(device='cpu', diarization=True, keep_model=True), lambda _: None, lambda: False,
        on_segment=emitted.append, speaker_tracker=FakeTracker())
    assert result['diarization'] is True
    assert [(s['speaker'], s['text']) for s in emitted] == [('Rozmówca 1', 'Pierwszy'), ('Rozmówca 2', 'Drugi')]
    assert emitted == result['segments'] and emitted[0]['start'] == 30.1


def test_corrected_speaker_is_persisted_and_exported(tmp_path):
    db = Database(tmp_path / 'jobs.sqlite')
    job = db.add_job('audio.wav', 'audio.wav', Settings())
    result = {'segments': [{'start': 0, 'end': 1, 'text': 'Witaj', 'speaker': 'Rozmówca 1'},
                           {'start': 1, 'end': 2, 'text': 'Cześć', 'speaker': None}]}
    db.update(job, status='done', result=json.dumps(result))
    db.edit_speaker(job, 0, '  Anna  ')
    saved = json.loads(Database(tmp_path / 'jobs.sqlite').get(job)['result'])
    assert saved['segments'][0]['speaker'] == 'Anna'
    assert render(saved, 'txt') == 'Anna: Witaj\nCześć\n'
    assert 'Anna: Witaj' in render(saved, 'srt')
    assert 'Anna: Witaj' in render(saved, 'vtt')
    assert 'start,end,speaker,text' in render(saved, 'csv')
    assert json.loads(render(saved, 'json'))['segments'][0]['speaker'] == 'Anna'
    db.edit_speaker(job, 0, '')
    assert json.loads(db.get(job)['result'])['segments'][0]['speaker'] is None
    with pytest.raises(ValueError):
        db.edit_speaker(job, 0, 'x' * 81)


def test_queue_keeps_speaker_labels_in_result_and_export(tmp_path, monkeypatch):
    class ReadyModels:
        def available(self, name, cache_dir):
            return True
    class FakeTracker:
        def __init__(self, cache_dir, cancelled):
            assert not cancelled()
    class FakeEngine:
        def transcribe(self, path, options, progress, cancelled, on_segment, on_started, speaker_tracker):
            assert isinstance(speaker_tracker, FakeTracker)
            on_started()
            item = {'start': 0., 'end': 1., 'text': 'Cześć', 'speaker': 'Rozmówca 1'}
            on_segment(item)
            progress(1.)
            return {'language': 'pl', 'segments': [item], 'diarization': True}
    monkeypatch.setattr(task_queue, 'SpeakerDiarizer', FakeTracker)
    db = Database(tmp_path / 'jobs.sqlite')
    fixture = Path(__file__).parent / 'fixtures' / 'polish.wav'
    options = Settings(model='tiny', device='cpu', diarization=True, cache_dir=str(tmp_path), output_dir=str(tmp_path))
    job_id = db.add_job('mowa.wav', str(fixture), options)
    queue = task_queue.TaskQueue(db, ReadyModels())
    queue._run(db.claim(), FakeEngine())
    job = db.get(job_id)
    assert job['status'] == 'done', job
    assert json.loads(job['result'])['segments'][0]['speaker'] == 'Rozmówca 1'
    assert db.live_segments(job_id)[0]['speaker'] == 'Rozmówca 1'
    assert (tmp_path / f'{job_id}.txt').read_text(encoding='utf-8').startswith('Rozmówca 1:')


@pytest.mark.skipif(os.environ.get('WHYSPER_SPEAKER_MODEL_TEST') != '1',
                    reason='Requires the locally downloaded speaker models')
def test_real_speaker_models_identify_two_voices_across_windows():
    root = Path(__file__).parent / 'fixtures'
    english = next(chunks(root / 'jfk.flac', seconds=5))[1]
    polish = next(chunks(root / 'polish.wav', seconds=5))[1]
    tracker = speakers.SpeakerDiarizer(Settings().cache_dir)
    def labels(samples):
        return {turn['speaker'] for turn in tracker.turns(samples) if turn['speaker']}
    first = labels(english)
    second = labels(polish)
    repeated = labels(english)
    assert first and second and first.isdisjoint(second)
    assert first & repeated
