import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import sounddevice as sd

from app.config import Settings
from app.core import live_captions, recorder
from app.core.live_captions import LiveCaptions, audio_16khz
from app.core.recorder import Recorder
from app.core.whisper_engine import Cancelled


def wait_for(predicate, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    assert predicate(), 'Timed out waiting for live captions'


class FakeModels:
    def __init__(self, available=True, downloading=False):
        self.ready, self.downloading = available, downloading

    def available(self, name, cache_dir):
        return self.ready

    def is_downloading(self, name, cache_dir):
        return self.downloading


class FakeEngine:
    def __init__(self):
        self.prepared = []
        self.calls = []
        self.unloaded = 0

    def prepare(self, options):
        self.prepared.append(options)

    def transcribe_samples(self, samples, options, cancelled):
        self.calls.append((samples.copy(), options))
        return {'language': 'pl', 'segments': [{'text': f'fragment {len(self.calls)}'}]}

    def unload(self):
        self.unloaded += 1


@pytest.fixture
def audio_input(monkeypatch):
    streams = []

    class FakeStream:
        def __init__(self, **kwargs):
            self.callback = kwargs['callback']
            self.rate = kwargs['samplerate']
            self.aborted = threading.Event()
            self.closed = threading.Event()
            streams.append(self)

        def start(self):
            pass

        def feed(self, samples, status=None):
            samples = np.asarray(samples, dtype=np.int16).reshape(-1, 1)
            self.callback(samples, len(samples), None, status)

        def abort(self):
            self.aborted.set()

        def stop(self):
            self.aborted.set()

        def close(self):
            self.closed.set()

    monkeypatch.setattr(sd, 'check_input_settings', lambda **kwargs: None)
    monkeypatch.setattr(sd, 'InputStream', FakeStream)
    return streams


def make_service(monkeypatch, tmp_path, *, device='cpu', models=None, engine=None):
    models = models or FakeModels()
    engine = engine or FakeEngine()
    monkeypatch.setattr(live_captions, 'WhisperEngine', lambda _: engine)
    tasks = SimpleNamespace(stop_event=threading.Event(), gpu_slot=threading.Lock(), gpu_engine=engine)
    microphone_lock = threading.Lock()
    service = LiveCaptions(models, tasks, microphone_lock)
    options = Settings(model='tiny', device=device, cache_dir=str(tmp_path), word_timestamps=True)
    return service, options, engine, tasks, microphone_lock


def test_microphone_captions_and_final_partial_chunk(monkeypatch, tmp_path, audio_input):
    service, options, engine, _, lock = make_service(monkeypatch, tmp_path)
    service.start(3, options, 2, 'tab-a')
    try:
        wait_for(lambda: service.snapshot()['phase'] == 'listening')
        assert not service.stop('other-tab')
        stream = audio_input[0]
        stream.feed(np.full(32000, 16384, dtype=np.int16))
        wait_for(lambda: service.transcript() == 'fragment 1')
        stream.feed(np.full(8000, 8192, dtype=np.int16))
        assert service.stop('tab-a', finish=True)
        assert stream.aborted.wait(2)
        wait_for(lambda: service.snapshot()['phase'] == 'stopped')
        assert service.transcript() == 'fragment 1\nfragment 2'
        assert len(engine.calls[0][0]) == 32000 and len(engine.calls[1][0]) == 8000
        assert engine.calls[0][1].language == '' and engine.calls[1][1].language == ''
        assert not engine.calls[0][1].word_timestamps
        assert lock.acquire(blocking=False)
        lock.release()
    finally:
        service.shutdown()


def test_resampling_and_device_default_rate_fallback(monkeypatch, tmp_path, audio_input):
    samples = (np.sin(np.arange(48000) * 2 * np.pi * 440 / 48000) * 16000).astype(np.int16)
    converted = audio_16khz(samples, 48000)
    assert converted.dtype == np.float32 and len(converted) == 16000
    assert np.max(np.abs(converted)) > .4

    def check(**kwargs):
        if kwargs['samplerate'] == 16000:
            raise sd.PortAudioError('unsupported rate')

    monkeypatch.setattr(sd, 'check_input_settings', check)
    monkeypatch.setattr(sd, 'query_devices', lambda device, kind: {'default_samplerate': 48000})
    service, options, engine, _, _ = make_service(monkeypatch, tmp_path)
    service.start(3, options, 2, 'tab-a')
    try:
        wait_for(lambda: service.snapshot()['phase'] == 'listening')
        assert audio_input[0].rate == 48000
        audio_input[0].feed(np.tile(samples, 2))
        wait_for(lambda: len(engine.calls) == 1)
        assert len(engine.calls[0][0]) == 32000
    finally:
        service.shutdown()


def test_backlog_is_bounded_and_reports_dropped_audio(monkeypatch, tmp_path, audio_input):
    entered = threading.Event()
    release = threading.Event()

    class SlowEngine(FakeEngine):
        def transcribe_samples(self, samples, options, cancelled):
            entered.set()
            while not release.wait(.02):
                if cancelled():
                    raise Cancelled()
            return super().transcribe_samples(samples, options, cancelled)

    service, options, _, _, _ = make_service(monkeypatch, tmp_path, engine=SlowEngine())
    service.start(3, options, 2, 'tab-a')
    try:
        wait_for(lambda: service.snapshot()['phase'] == 'listening')
        stream = audio_input[0]
        chunk = np.full(32000, 1000, dtype=np.int16)
        stream.feed(chunk)
        assert entered.wait(3)
        for _ in range(8):
            stream.feed(chunk)
        snapshot = service.snapshot()
        assert snapshot['pending'] <= service.MAX_PENDING
        assert snapshot['dropped_seconds'] > 0
    finally:
        release.set()
        service.shutdown()


def test_stop_closes_microphone_before_current_inference_finishes(monkeypatch, tmp_path, audio_input):
    entered = threading.Event()
    release = threading.Event()

    class SlowEngine(FakeEngine):
        def transcribe_samples(self, samples, options, cancelled):
            entered.set()
            assert release.wait(5)
            return super().transcribe_samples(samples, options, cancelled)

    service, options, _, _, _ = make_service(monkeypatch, tmp_path, engine=SlowEngine())
    service.start(3, options, 2, 'tab-a')
    try:
        wait_for(lambda: service.snapshot()['phase'] == 'listening')
        stream = audio_input[0]
        stream.feed(np.ones(32000, dtype=np.int16))
        assert entered.wait(3)
        assert service.stop('tab-a', finish=True)
        assert stream.aborted.wait(2)
        assert service.snapshot()['phase'] == 'stopping'
        release.set()
        wait_for(lambda: service.snapshot()['phase'] == 'stopped')
        assert service.transcript() == 'fragment 1'
    finally:
        release.set()
        service.shutdown()


def test_gpu_slot_and_microphone_are_released_when_session_stops(monkeypatch, tmp_path, audio_input):
    service, options, engine, tasks, microphone_lock = make_service(monkeypatch, tmp_path, device='cuda')
    tasks.gpu_slot.acquire()
    owned_by_test = True
    service.start(3, options, 2, 'tab-a')
    try:
        wait_for(lambda: service.snapshot()['phase'] == 'waiting_gpu')
        assert not audio_input
        tasks.gpu_slot.release()
        owned_by_test = False
        wait_for(lambda: service.snapshot()['phase'] == 'listening')
        assert service.using_model('tiny', str(tmp_path))
        assert not microphone_lock.acquire(blocking=False)
        assert service.stop('tab-a', finish=False)
        wait_for(lambda: service.snapshot()['phase'] == 'stopped')
        assert tasks.gpu_slot.acquire(blocking=False)
        tasks.gpu_slot.release()
        assert microphone_lock.acquire(blocking=False)
        microphone_lock.release()
        assert engine.unloaded == 1
    finally:
        if owned_by_test:
            tasks.gpu_slot.release()
        service.shutdown()


def test_vulkan_live_captions_use_vulkan_model_and_gpu_slot(monkeypatch, tmp_path, audio_input):
    class VulkanModels:
        def is_downloading(self, name, cache_dir, device):
            assert (name, device) == ('tiny', 'vulkan')
            return False

        def available(self, name, cache_dir, device):
            assert (name, device) == ('tiny', 'vulkan')
            return True

    service, options, engine, tasks, _ = make_service(monkeypatch, tmp_path, device='vulkan', models=VulkanModels())
    tasks.vulkan_engine = engine
    service.start(3, options, 2, 'tab-a')
    try:
        wait_for(lambda: service.snapshot()['phase'] == 'listening')
        assert not tasks.gpu_slot.acquire(blocking=False)
        audio_input[0].feed(np.full(32000, 16384, dtype=np.int16))
        wait_for(lambda: service.transcript() == 'fragment 1')
        service.stop('tab-a', finish=False)
        wait_for(lambda: service.snapshot()['phase'] == 'stopped')
        assert tasks.gpu_slot.acquire(blocking=False)
        tasks.gpu_slot.release()
        assert engine.unloaded == 1
    finally:
        service.shutdown()


def test_recording_and_live_captions_do_not_open_the_microphone_together(monkeypatch, tmp_path, audio_input):
    service, options, _, _, microphone_lock = make_service(monkeypatch, tmp_path)
    monkeypatch.setattr(recorder, 'RECORDINGS', tmp_path / 'recordings')
    recording = Recorder(microphone_lock)
    service.start(3, options, 2, 'tab-a')
    try:
        wait_for(lambda: service.snapshot()['phase'] == 'listening')
        with pytest.raises(RuntimeError, match='Mikrofon jest już używany'):
            recording.start(3)
        service.shutdown()
        recording.start(3)
        assert recording.status()['active']
        recording.stop()
        assert not recording.status()['active']
    finally:
        service.shutdown()
        if recording.status()['active']:
            recording.stop()


def test_unload_failure_still_releases_gpu(monkeypatch, tmp_path, audio_input):
    class BrokenUnload(FakeEngine):
        def unload(self):
            raise RuntimeError('could not unload model')

    service, options, _, tasks, microphone_lock = make_service(monkeypatch, tmp_path, device='cuda', engine=BrokenUnload())
    service.start(3, options, 2, 'tab-a')
    try:
        wait_for(lambda: service.snapshot()['phase'] == 'listening')
        assert service.stop('tab-a', finish=False)
        wait_for(lambda: service.snapshot()['phase'] == 'error')
        assert 'zwolnić modelu' in service.snapshot()['error']
        assert tasks.gpu_slot.acquire(blocking=False)
        tasks.gpu_slot.release()
        assert microphone_lock.acquire(blocking=False)
        microphone_lock.release()
    finally:
        service.shutdown()


def test_microphone_close_failure_is_reported(monkeypatch, tmp_path, audio_input):
    service, options, _, _, microphone_lock = make_service(monkeypatch, tmp_path)
    service.start(3, options, 2, 'tab-a')
    try:
        wait_for(lambda: service.snapshot()['phase'] == 'listening')
        def fail_close():
            raise OSError('cannot close audio input')
        audio_input[0].close = fail_close
        assert service.stop('tab-a', finish=False)
        wait_for(lambda: service.snapshot()['phase'] == 'error')
        assert 'zamknąć mikrofonu' in service.snapshot()['error']
        assert microphone_lock.acquire(blocking=False)
        microphone_lock.release()
    finally:
        service.shutdown()


def test_model_validation_does_not_open_the_microphone(monkeypatch, tmp_path, audio_input):
    service, options, _, _, _ = make_service(monkeypatch, tmp_path, models=FakeModels(available=False))
    with pytest.raises(ValueError, match='Najpierw pobierz'):
        service.start(3, options, 5, 'tab-a')
    service.models = FakeModels(downloading=True)
    with pytest.raises(ValueError, match='Poczekaj'):
        service.start(3, options, 5, 'tab-a')
    service.models = FakeModels()
    with pytest.raises(ValueError, match='transkrypcję'):
        service.start(3, options.model_copy(update={'task': 'translate'}), 5, 'tab-a')
    assert not audio_input


def test_stop_while_model_is_loading_never_opens_microphone(monkeypatch, tmp_path, audio_input):
    entered = threading.Event()
    release = threading.Event()

    class SlowPrepare(FakeEngine):
        def prepare(self, options):
            entered.set()
            assert release.wait(5)

    service, options, _, _, _ = make_service(monkeypatch, tmp_path, engine=SlowPrepare())
    service.start(3, options, 2, 'tab-a')
    try:
        assert entered.wait(3)
        assert service.stop('tab-a', finish=False)
        release.set()
        wait_for(lambda: service.snapshot()['phase'] == 'stopped')
        assert not audio_input
    finally:
        release.set()
        service.shutdown()


def test_microphone_error_does_not_report_a_cuda_failure(monkeypatch, tmp_path, audio_input):
    monkeypatch.setattr(sd, 'InputStream', lambda **kwargs: (_ for _ in ()).throw(sd.PortAudioError('Invalid device')))
    service, options, _, _, microphone_lock = make_service(monkeypatch, tmp_path)
    service.start(3, options, 2, 'tab-a')
    try:
        wait_for(lambda: service.snapshot()['phase'] == 'error')
        assert 'mikrofonu' in service.snapshot()['error']
        assert 'CUDA' not in service.snapshot()['error']
        assert microphone_lock.acquire(blocking=False)
        microphone_lock.release()
    finally:
        service.shutdown()


def test_disconnected_microphone_is_reported_and_released(monkeypatch, tmp_path, audio_input):
    service, options, _, _, microphone_lock = make_service(monkeypatch, tmp_path)
    service.start(3, options, 2, 'tab-a')
    try:
        wait_for(lambda: service.snapshot()['phase'] == 'listening')
        audio_input[0].active = False
        wait_for(lambda: service.snapshot()['phase'] == 'error')
        assert 'Strumień mikrofonu został przerwany' in service.snapshot()['error']
        assert audio_input[0].closed.wait(2)
        assert microphone_lock.acquire(blocking=False)
        microphone_lock.release()
    finally:
        service.shutdown()


def test_silence_does_not_fix_the_detected_language(monkeypatch, tmp_path, audio_input):
    class SilentFirst(FakeEngine):
        def transcribe_samples(self, samples, options, cancelled):
            self.calls.append((samples.copy(), options))
            if len(self.calls) == 1:
                return {'language': 'en', 'segments': []}
            return {'language': 'pl', 'segments': [{'text': 'mowa'}]}

    service, options, engine, _, _ = make_service(monkeypatch, tmp_path, engine=SilentFirst())
    service.start(3, options, 2, 'tab-a')
    try:
        wait_for(lambda: service.snapshot()['phase'] == 'listening')
        audio_input[0].feed(np.zeros(32000, dtype=np.int16))
        wait_for(lambda: len(engine.calls) == 1)
        assert service.snapshot()['language'] is None
        audio_input[0].feed(np.ones(32000, dtype=np.int16))
        wait_for(lambda: service.transcript() == 'mowa')
        assert engine.calls[1][1].language == ''
        assert service.snapshot()['language'] == 'pl'
    finally:
        service.shutdown()


def test_automatic_language_is_detected_again_for_each_live_chunk(monkeypatch, tmp_path, audio_input):
    class DifferentLanguages(FakeEngine):
        def transcribe_samples(self, samples, options, cancelled):
            self.calls.append((samples.copy(), options))
            language = 'en' if len(self.calls) == 1 else 'pl'
            return {'language': language, 'segments': [{'text': 'hello' if language == 'en' else 'cześć'}]}

    service, options, engine, _, _ = make_service(monkeypatch, tmp_path, engine=DifferentLanguages())
    service.start(3, options, 2, 'tab-a')
    try:
        wait_for(lambda: service.snapshot()['phase'] == 'listening')
        audio_input[0].feed(np.ones(32000, dtype=np.int16))
        wait_for(lambda: service.transcript() == 'hello')
        audio_input[0].feed(np.ones(32000, dtype=np.int16))
        wait_for(lambda: service.transcript() == 'hello\ncześć')
        assert [call[1].language for call in engine.calls] == ['', '']
        assert service.snapshot()['language'] == 'pl'
    finally:
        service.shutdown()


def test_live_captions_show_and_keep_speaker_labels(monkeypatch, tmp_path, audio_input):
    class TimedEngine(FakeEngine):
        def transcribe_samples(self, samples, options, cancelled):
            assert options.word_timestamps
            return {'language': 'pl', 'segments': [{'start': 0., 'end': 2., 'text': 'Cześć Witaj', 'words': [
                {'start': .1, 'end': .8, 'word': ' Cześć'},
                {'start': 1.1, 'end': 1.8, 'word': ' Witaj'}]}]}

    class FakeTracker:
        def __init__(self, cache_dir, cancelled):
            self.calls = 0

        def turns(self, samples, cancelled):
            self.calls += 1
            return [{'start': 0, 'end': 1, 'speaker': 'Rozmówca 1'},
                    {'start': 1, 'end': 2, 'speaker': 'Rozmówca 2'}]

    monkeypatch.setattr(live_captions, 'SpeakerDiarizer', FakeTracker)
    service, options, _, _, _ = make_service(monkeypatch, tmp_path, engine=TimedEngine())
    service.start(3, options.model_copy(update={'diarization': True}), 2, 'tab-a')
    try:
        wait_for(lambda: service.snapshot()['phase'] == 'listening')
        audio_input[0].feed(np.ones(32000, dtype=np.int16))
        wait_for(lambda: service.transcript() == 'Rozmówca 1: Cześć\nRozmówca 2: Witaj')
        audio_input[0].feed(np.ones(32000, dtype=np.int16))
        wait_for(lambda: service.transcript().count('Rozmówca 1') == 2)
    finally:
        service.shutdown()


def test_stopping_during_speaker_preparation_never_opens_microphone(monkeypatch, tmp_path, audio_input):
    entered = threading.Event()

    class SlowTracker:
        def __init__(self, cache_dir, cancelled):
            entered.set()
            while not cancelled():
                time.sleep(.01)
            raise live_captions.SpeakerCancelled()

    monkeypatch.setattr(live_captions, 'SpeakerDiarizer', SlowTracker)
    service, options, _, _, _ = make_service(monkeypatch, tmp_path)
    service.start(3, options.model_copy(update={'diarization': True}), 2, 'tab-a')
    try:
        assert entered.wait(3)
        assert service.stop('tab-a', finish=False)
        wait_for(lambda: service.snapshot()['phase'] == 'stopped')
        assert not audio_input
    finally:
        service.shutdown()
