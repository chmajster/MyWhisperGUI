import asyncio
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from nicegui import ui

from app.config import Settings
from app.core import audio_processor, gpu_manager, gpu_probe, whisper_engine
from app.core.whisper_engine import WhisperEngine
from app.database.db import Database
from app.ui import layout


GPU_A = 'GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
GPU_B = 'GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'


@pytest.fixture
def fake_cuda(monkeypatch):
    # CUDA order is B, A; it is deliberately different from physical NVML order.
    properties = [
        SimpleNamespace(uuid=GPU_B[4:], name='Test GPU B', total_memory=24 * 1024**3, major=8, minor=9),
        SimpleNamespace(uuid=GPU_A[4:], name='Test GPU A', total_memory=8 * 1024**3, major=8, minor=6),
    ]
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2,
                                                get_device_properties=lambda index: properties[index],
                                                mem_get_info=lambda index: (10 + index, 100 + index)))
    ct2 = SimpleNamespace(get_cuda_device_count=lambda: 2,
                          get_supported_compute_types=lambda device, index: {'float16', 'int8_float16'})
    nvml_calls = []

    class NvmlError(Exception):
        pass

    class FakeNvml:
        NVMLError = NvmlError
        NVML_TEMPERATURE_GPU = 0

        @staticmethod
        def nvmlInit():
            pass

        @staticmethod
        def nvmlShutdown():
            pass

        @staticmethod
        def nvmlDeviceGetHandleByUUID(uuid):
            nvml_calls.append(uuid)
            return uuid

        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle):
            return SimpleNamespace(total=200 if handle == GPU_B else 100,
                                   used=70 if handle == GPU_B else 10)

        @staticmethod
        def nvmlSystemGetDriverVersion():
            return 'test-driver'

        @staticmethod
        def nvmlDeviceGetUtilizationRates(handle):
            return SimpleNamespace(gpu=42 if handle == GPU_B else 11)

        @staticmethod
        def nvmlDeviceGetTemperature(handle, sensor):
            return 55 if handle == GPU_B else 44

    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'ctranslate2', ct2)
    monkeypatch.setitem(sys.modules, 'pynvml', FakeNvml)
    monkeypatch.setattr(gpu_manager, 'configure_cuda_dlls', lambda: [])
    return properties, nvml_calls


def test_gpu_detection_uses_cuda_order_and_nvml_uuid(fake_cuda):
    _, nvml_calls = fake_cuda
    devices = gpu_manager.cuda_devices()
    assert [(item['index'], item['id']) for item in devices] == [(0, GPU_B), (1, GPU_A)]
    assert gpu_manager.resolve_cuda_device(GPU_A, devices)['index'] == 1
    stats = gpu_manager.gpu_stats(GPU_B)
    assert stats['index'] == 0 and stats['name'] == 'Test GPU B'
    assert stats['memory_total'] == 200 and stats['memory_used'] == 70
    assert stats['utilization'] == 42 and stats['temperature'] == 55
    assert nvml_calls == [GPU_B]


def test_uuid_selection_follows_reordered_cards_and_rejects_missing(fake_cuda):
    properties, _ = fake_cuda
    assert gpu_manager.resolve_cuda_device(GPU_B)['index'] == 0
    properties.reverse()
    assert gpu_manager.resolve_cuda_device(GPU_B)['index'] == 1
    assert gpu_manager.resolve_cuda_device('cuda:0')['id'] == GPU_A
    with pytest.raises(RuntimeError, match='nie jest dostępna'):
        gpu_manager.resolve_cuda_device('GPU-missing')


def test_cuda_device_ordinals_remain_selectable_when_metadata_fails(fake_cuda, monkeypatch):
    torch = sys.modules['torch']
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: (_ for _ in ()).throw(RuntimeError('no metadata')))
    devices = gpu_manager.cuda_devices()
    assert [item['id'] for item in devices] == ['cuda:0', 'cuda:1']
    stats = gpu_manager.gpu_stats('cuda:1')
    assert stats['memory_total'] == 101 and stats['memory_used'] == 90
    assert stats['utilization'] is None


def test_nvml_failure_uses_the_selected_cuda_memory(fake_cuda, monkeypatch):
    nvml = sys.modules['pynvml']
    monkeypatch.setattr(nvml, 'nvmlDeviceGetHandleByUUID', lambda _: (_ for _ in ()).throw(RuntimeError('NVML unavailable')))
    stats = gpu_manager.gpu_stats(GPU_A)
    assert stats['index'] == 1
    assert stats['memory_total'] == 101 and stats['memory_used'] == 90


def test_no_cuda_device_is_not_confused_with_a_physical_nvml_gpu(fake_cuda, monkeypatch):
    monkeypatch.setattr(sys.modules['ctranslate2'], 'get_cuda_device_count', lambda: 0)
    assert gpu_manager.cuda_devices() == []
    assert 'Nie wykryto' in gpu_manager.gpu_stats()['error']


def test_engine_uses_selected_gpu_and_reloads_when_device_changes(tmp_path, monkeypatch):
    selected = {
        GPU_A: {'id': GPU_A, 'index': 0, 'name': 'GPU A'},
        GPU_B: {'id': GPU_B, 'index': 1, 'name': 'GPU B'},
    }
    selected_stats = {GPU_A: {'memory_total': 8 * 1024**3, 'memory_used': 1024**3},
                      GPU_B: {'memory_total': 8 * 1024**3, 'memory_used': int(7.5 * 1024**3)}}
    calls, probes, unloaded = [], [], []

    class FakeWhisperModel:
        def __init__(self, path, **kwargs):
            calls.append(kwargs)
            self.model = SimpleNamespace(device=kwargs['device'], device_index=[kwargs['device_index']],
                                         compute_type=kwargs['compute_type'],
                                         unload_model=lambda **kw: unloaded.append(kwargs['device_index']))

    ct2 = SimpleNamespace(get_supported_compute_types=lambda device, index: probes.append(index) or {'float16', 'int8_float16'})
    monkeypatch.setitem(sys.modules, 'ctranslate2', ct2)
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setattr(whisper_engine, 'configure_cuda_dlls', lambda: [])
    monkeypatch.setattr(whisper_engine, 'check_cuda_libraries', lambda: None)
    monkeypatch.setattr(whisper_engine, 'resolve_cuda_device', lambda gpu_id: selected[gpu_id])
    monkeypatch.setattr(whisper_engine, 'gpu_stats', lambda gpu_id: selected_stats[gpu_id])
    models = SimpleNamespace(path=lambda name, directory: tmp_path / name)
    engine = WhisperEngine(models)
    a = Settings(model='tiny', device='cuda', gpu_id=GPU_A)
    b = Settings(model='tiny', device='cuda', gpu_id=GPU_B)
    first, precision = engine._model(a)
    assert precision == 'float16' and calls[-1]['device_index'] == 0
    assert engine._model(a)[0] is first and len(calls) == 1
    second, precision = engine._model(b)
    assert second is not first and precision == 'int8_float16'
    assert calls[-1]['device_index'] == 1 and unloaded == [0]
    assert probes[-1] == 1 and engine.loaded_gpu['id'] == GPU_B


def test_engine_rejects_a_backend_on_the_wrong_gpu(tmp_path, monkeypatch):
    class WrongGpuModel:
        def __init__(self, path, **kwargs):
            self.model = SimpleNamespace(device='cuda', device_index=[0], compute_type='float16',
                                         unload_model=lambda **kw: None)

    monkeypatch.setitem(sys.modules, 'ctranslate2', SimpleNamespace(get_supported_compute_types=lambda *args: {'float16'}))
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=WrongGpuModel))
    monkeypatch.setattr(whisper_engine, 'configure_cuda_dlls', lambda: [])
    monkeypatch.setattr(whisper_engine, 'check_cuda_libraries', lambda: None)
    monkeypatch.setattr(whisper_engine, 'resolve_cuda_device', lambda _: {'id': GPU_B, 'index': 1, 'name': 'GPU B'})
    engine = WhisperEngine(SimpleNamespace(path=lambda name, directory: tmp_path / name))
    with pytest.raises(RuntimeError, match='innej karcie GPU'):
        engine._model(Settings(model='tiny', gpu_id=GPU_B, compute_type='float16'))
    assert engine.model is None


def test_auto_precision_uses_a_supported_fallback_on_the_selected_gpu(tmp_path, monkeypatch):
    calls = []

    class FakeWhisperModel:
        def __init__(self, path, **kwargs):
            calls.append(kwargs)
            self.model = SimpleNamespace(device='cuda', device_index=[kwargs['device_index']],
                                         compute_type=kwargs['compute_type'], unload_model=lambda **kw: None)

    monkeypatch.setitem(sys.modules, 'ctranslate2', SimpleNamespace(get_supported_compute_types=lambda *args: {'int8_float32'}))
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setattr(whisper_engine, 'configure_cuda_dlls', lambda: [])
    monkeypatch.setattr(whisper_engine, 'check_cuda_libraries', lambda: None)
    monkeypatch.setattr(whisper_engine, 'resolve_cuda_device', lambda _: {'id': GPU_B, 'index': 1, 'name': 'GPU B'})
    monkeypatch.setattr(whisper_engine, 'gpu_stats', lambda _: {'memory_total': 8 * 1024**3, 'memory_used': 1024**3})
    engine = WhisperEngine(SimpleNamespace(path=lambda name, directory: tmp_path / name))
    _, precision = engine._model(Settings(model='tiny', gpu_id=GPU_B))
    assert precision == 'int8_float32'
    assert calls[0]['device_index'] == 1 and calls[0]['compute_type'] == 'int8_float32'


def test_transcription_result_records_the_used_gpu(monkeypatch):
    class FakeModel:
        def transcribe(self, samples, **kwargs):
            segment = SimpleNamespace(start=0, end=1, text='test', words=None)
            return iter([segment]), SimpleNamespace(language='pl', language_probability=.9)

    def load(engine, options):
        engine.loaded_gpu = {'id': GPU_B, 'index': 1, 'name': 'GPU B'}
        return FakeModel(), 'float16'

    monkeypatch.setattr(WhisperEngine, '_model', load)
    monkeypatch.setattr(audio_processor, 'chunks', lambda path: iter([(0, np.zeros(16000))]))
    options = Settings(device='cuda', gpu_id=GPU_B, keep_model=True)
    result = WhisperEngine(None).transcribe('unused.wav', options, lambda _: None, lambda: False)
    assert result['gpu_id'] == GPU_B and result['gpu_index'] == 1 and result['gpu_name'] == 'GPU B'


def test_gpu_selection_persists_in_settings_and_jobs(tmp_path):
    assert Settings.model_validate_json('{"device":"cuda"}').gpu_id == ''
    db = Database(tmp_path / 'gpu.sqlite')
    settings = Settings(device='cuda', gpu_id=GPU_B)
    db.save_settings(settings)
    job_id = db.add_job('sample.wav', 'sample.wav', settings)
    assert db.settings().gpu_id == GPU_B
    assert Settings.model_validate_json(db.get(job_id)['options']).gpu_id == GPU_B


def test_dashboard_monitors_active_gpu_instead_of_a_parallel_cpu_job():
    default = Settings(gpu_id=GPU_A)
    cpu = {'status': 'transcribing', 'options': Settings(device='cpu').model_dump_json()}
    preparing = {'status': 'preparing', 'options': Settings(gpu_id=GPU_A).model_dump_json()}
    active = {'status': 'transcribing', 'options': Settings(gpu_id=GPU_B).model_dump_json()}
    assert layout.dashboard_gpu_id([cpu, preparing, active], default, None) == GPU_B
    assert layout.dashboard_gpu_id([cpu], default, {'id': GPU_B}) == GPU_B
    assert layout.dashboard_gpu_id([cpu], default, None) == GPU_A


@pytest.mark.parametrize('selected,missing', [(GPU_B, False), ('GPU-missing', True), ('cuda:1', False)])
def test_gpu_picker_lists_cards_and_preserves_selection(monkeypatch, selected, missing):
    callbacks = []
    devices = [{'id': GPU_A, 'index': 0, 'name': 'GPU A', 'memory_total': 8 * 1024**3},
               {'id': GPU_B, 'index': 1, 'name': 'GPU B', 'memory_total': 24 * 1024**3}]
    monkeypatch.setattr(layout, 'cuda_devices', lambda: devices)
    monkeypatch.setattr(ui, 'timer', lambda _, callback, **kwargs: callbacks.append(callback))
    with ui.card() as panel:
        fields = layout.option_form(Settings(gpu_id=selected))
    try:
        assert not fields['gpu_id'].enabled
        asyncio.run(callbacks[0]())
        picker = fields['gpu_id']
        assert picker.enabled and GPU_A in picker.options and GPU_B in picker.options
        assert picker.value == selected
        if missing:
            assert 'Niedostępna karta' in picker.options[selected]
            assert any('Zapisana karta nie jest dostępna' in (getattr(item, 'text', '') or '') for item in panel.descendants())
            picker.value = GPU_A
            assert not any('Zapisana karta nie jest dostępna' in (getattr(item, 'text', '') or '') for item in panel.descendants())
            picker.value = selected
        elif selected == 'cuda:1':
            assert 'zapis według indeksu' in picker.options[selected]
        else:
            assert 'CUDA 1' in picker.options[GPU_B] and 'GPU B' in picker.options[GPU_B]
        fields['device'].value = 'cpu'
        assert not picker.enabled
        fields['device'].value = 'cuda'
        assert picker.enabled
        assert layout.form_values(fields, Settings()).gpu_id == selected
    finally:
        panel.delete()


def test_gpu_picker_preserves_missing_selection_when_no_cuda(monkeypatch):
    callbacks = []
    monkeypatch.setattr(layout, 'cuda_devices', lambda: [])
    monkeypatch.setattr(ui, 'timer', lambda _, callback, **kwargs: callbacks.append(callback))
    with ui.card() as panel:
        fields = layout.option_form(Settings(gpu_id=GPU_B))
    try:
        asyncio.run(callbacks[0]())
        picker = fields['gpu_id']
        assert not picker.enabled and picker.value == GPU_B
        assert 'Niedostępna karta' in picker.options[GPU_B]
        assert any('Zapisana karta nie jest dostępna' in (getattr(item, 'text', '') or '') for item in panel.descendants())
        fields['device'].value = 'cpu'
        assert layout.form_values(fields, Settings()).gpu_id == GPU_B
    finally:
        panel.delete()


def test_inference_probe_passes_gpu_identity_to_child(monkeypatch, tmp_path):
    captured = []

    def run(command, **kwargs):
        captured.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({'ok': True, 'gpu_index': 1}), stderr='')

    monkeypatch.setattr(gpu_manager.subprocess, 'run', run)
    result = gpu_manager.inference_test(tmp_path / 'model', tmp_path / 'audio.wav', gpu_id=GPU_B)
    assert result['gpu_index'] == 1
    assert captured[0][-2:] == ['--gpu-id', GPU_B]
    assert str(tmp_path / 'audio.wav') in captured[0]


def test_gpu_probe_uses_requested_device(monkeypatch, capsys):
    calls = []

    class FakeWhisperModel:
        def __init__(self, path, **kwargs):
            calls.append(kwargs)
            self.model = SimpleNamespace(device='cuda', device_index=[kwargs['device_index']], compute_type='float16')

        def transcribe(self, audio, **kwargs):
            return iter([SimpleNamespace(text='test')]), SimpleNamespace(language='pl')

    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setattr(gpu_probe, 'configure_cuda_dlls', lambda: [])
    monkeypatch.setattr(gpu_probe, 'check_cuda_libraries', lambda: None)
    monkeypatch.setattr(gpu_probe, 'resolve_cuda_device', lambda gpu_id: {'id': gpu_id, 'index': 1, 'name': 'GPU B'})
    monkeypatch.setattr(gpu_probe, 'gpu_stats', lambda gpu_id: {'memory_used': 10})
    monkeypatch.setattr(sys, 'argv', ['gpu_probe', 'model', '--gpu-id', GPU_B])
    gpu_probe.main()
    result = json.loads(capsys.readouterr().out)
    assert calls[0]['device_index'] == 1
    assert result['gpu_id'] == GPU_B and result['gpu_index'] == 1 and result['ok']
