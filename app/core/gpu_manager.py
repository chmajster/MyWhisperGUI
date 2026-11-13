"""Windows CUDA setup and diagnostics. A positive probe means real inference."""
from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

from app.config import ROOT

_dll_handles = {}


def configure_cuda_dlls():
    """Make the CUDA 12/cuDNN 9 DLLs bundled with the Windows Torch wheel visible to CT2."""
    spec = importlib.util.find_spec('torch')
    paths = []
    if spec and spec.submodule_search_locations:
        paths.append(Path(next(iter(spec.submodule_search_locations))) / 'lib')
    paths.append(ROOT / 'cuda' / 'bin')
    paths = [p for p in paths if p.is_dir()]
    existing = os.environ.get('PATH', '').split(os.pathsep)
    os.environ['PATH'] = os.pathsep.join([*(str(p) for p in paths if str(p) not in existing), *existing])
    if hasattr(os, 'add_dll_directory'):
        for path in paths:
            if str(path) in _dll_handles:
                continue
            try:
                _dll_handles[str(path)] = os.add_dll_directory(str(path))
            except OSError:
                continue
    return paths


def check_cuda_libraries():
    configure_cuda_dlls()
    if os.name == 'nt':
        for name in ('cublas64_12.dll', 'cublasLt64_12.dll', 'cudnn64_9.dll', 'cudnn_ops64_9.dll', 'cudnn_cnn64_9.dll'):
            try:
                ctypes.WinDLL(name)
            except OSError as exc:
                raise RuntimeError(f'Brak wymaganej biblioteki {name}: {exc}. Uruchom repair.ps1.') from exc


def _gpu_uuid(value):
    if value is None:
        return None
    value = str(value).strip()
    if value.startswith(('GPU-', 'MIG-')):
        return value
    if re.fullmatch(r'[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}', value):
        return 'GPU-' + value
    return None


def cuda_devices():
    """List CUDA ordinals as seen by CTranslate2, with stable UUIDs when available."""
    configure_cuda_dlls()
    import ctranslate2
    count = ctranslate2.get_cuda_device_count()
    if not count:
        return []
    try:
        import torch
        torch_count = torch.cuda.device_count()
    except Exception:
        torch = None
        torch_count = 0
    devices = []
    for index in range(count):
        device = {'index': index, 'id': f'cuda:{index}', 'name': f'GPU CUDA {index}',
                  'memory_total': None, 'capability': None}
        if torch is not None and index < torch_count:
            try:
                properties = torch.cuda.get_device_properties(index)
                device['id'] = _gpu_uuid(getattr(properties, 'uuid', None)) or device['id']
                device['name'] = str(properties.name)
                device['memory_total'] = int(properties.total_memory)
                device['capability'] = f'{properties.major}.{properties.minor}'
            except Exception:
                pass  # Keep the CUDA ordinal selectable when metadata is unavailable.
        devices.append(device)
    return devices


def resolve_cuda_device(gpu_id='', devices=None):
    devices = cuda_devices() if devices is None else devices
    if not devices:
        raise RuntimeError('Nie wykryto urządzenia CUDA. Sprawdź sterownik NVIDIA lub wybierz CPU.')
    if not gpu_id:
        return devices[0]
    selected = next((device for device in devices if device['id'] == gpu_id), None)
    if selected is None and re.fullmatch(r'cuda:\d+', gpu_id):
        index = int(gpu_id[5:])
        selected = next((device for device in devices if device['index'] == index), None)
    if selected is None:
        raise RuntimeError(f'Wybrana karta GPU ({gpu_id}) nie jest dostępna. Wybierz inną kartę w ustawieniach.')
    return selected


def gpu_stats(gpu_id=''):
    try:
        device = resolve_cuda_device(gpu_id)
    except Exception as exc:
        return {'error': str(exc)}
    result = {**device, 'memory_used': None, 'utilization': None, 'temperature': None}
    try:
        import torch
        free, total = torch.cuda.mem_get_info(device['index'])
        result.update(memory_total=total, memory_used=total - free)
    except Exception:
        pass
    # NVML's physical indexes do not necessarily match CUDA's visible ordinals.
    # Only use a UUID match; never guess that the two indexes are equal.
    if device['id'].startswith(('GPU-', 'MIG-')):
        try:
            import pynvml as nv
            nv.nvmlInit()
            try:
                handle = nv.nvmlDeviceGetHandleByUUID(device['id'])
                mem = nv.nvmlDeviceGetMemoryInfo(handle)
                result.update(memory_total=mem.total, memory_used=mem.used, driver=nv.nvmlSystemGetDriverVersion())
                try:
                    result['utilization'] = nv.nvmlDeviceGetUtilizationRates(handle).gpu
                except nv.NVMLError:
                    pass
                try:
                    result['temperature'] = nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)
                except nv.NVMLError:
                    pass
            finally:
                nv.nvmlShutdown()
        except Exception:
            pass  # CUDA memory stats remain available even when NVML cannot map a GPU.
    return result


def diagnostics(gpu_id=''):
    paths = configure_cuda_dlls()
    result = {'python': platform.python_version(), 'gpu': gpu_stats(gpu_id), 'dll_paths': [str(p) for p in paths]}
    for package in ('ctranslate2', 'faster-whisper', 'torch'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = 'nie zainstalowano'
    for name in ('cublas64_12.dll', 'cudnn64_9.dll'):
        try:
            ctypes.WinDLL(name) if os.name == 'nt' else ctypes.CDLL(name)
            result[name] = 'dostępna'
        except OSError as exc:
            result[name] = f'brak lub błąd ładowania: {exc}'
    try:
        import ctranslate2
        devices = cuda_devices()
        result['cuda_devices'] = len(devices)
        result['cuda_gpus'] = devices
        if devices:
            selected = resolve_cuda_device(gpu_id, devices)
            result['compute_types'] = sorted(ctranslate2.get_supported_compute_types('cuda', selected['index']))
    except Exception as exc:
        result['cuda_error'] = str(exc)
    try:
        import torch
        result['torch_cuda'] = torch.version.cuda
        result['torch_cuda_available'] = torch.cuda.is_available()
        if result['torch_cuda_available']:
            selected = resolve_cuda_device(gpu_id)
            result['capability'] = '.'.join(map(str, torch.cuda.get_device_capability(selected['index'])))
            result['torch_architectures'] = torch.cuda.get_arch_list()
    except Exception as exc:
        result['torch_error'] = str(exc)
    return result


def friendly_error(exc):
    message = str(exc)
    lower = message.lower()
    if 'out of memory' in lower or 'cublas_status_alloc_failed' in lower:
        return 'Brak pamięci VRAM. Zamknij inne aplikacje GPU, wyłącz zatrzymywanie modelu albo wybierz mniejszy model. ' + message
    if any(word in lower for word in ('cublas', 'cudnn', 'dll', 'library')):
        return 'Nie można załadować bibliotek CUDA/cuBLAS/cuDNN. Uruchom repair.ps1 i sprawdź diagnostykę. ' + message
    if any(word in lower for word in ('cuda', 'driver', 'device')):
        return 'CUDA nie działa poprawnie. Sprawdź sterownik NVIDIA i uruchom test GPU w Diagnostyce. ' + message
    return message


def inference_test(model_dir, audio_path=None, timeout=300, *, gpu_id=''):
    """Isolate native failures and confirm Faster-Whisper is actually run on CUDA."""
    command = [sys.executable, '-m', 'app.core.gpu_probe', str(model_dir)]
    sample = ROOT / 'tests' / 'fixtures' / 'polish.wav'
    if audio_path is None and sample.is_file():
        audio_path = sample
    if audio_path:
        command.append(str(audio_path))
    command.extend(['--gpu-id', gpu_id])
    try:
        environment = os.environ.copy()
        environment['PYTHONIOENCODING'] = 'utf-8'
        proc = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
                              timeout=timeout, cwd=ROOT, env=environment)
        if proc.returncode:
            return {'ok': False, 'error': friendly_error((proc.stderr or proc.stdout or f'kod {proc.returncode}')[-3000:])}
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {'ok': False, 'error': friendly_error(exc)}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--gpu-id', default='', help='UUID GPU lub indeks CUDA, np. cuda:1')
    args = parser.parse_args()
    data = diagnostics(args.gpu_id)
    if args.test:
        from app.core.model_manager import ModelManager
        manager = ModelManager()
        try:
            if args.download and not manager.available('tiny', ROOT / 'models'):
                manager.download('tiny', ROOT / 'models')
            data['test_inferencji'] = inference_test(manager.path('tiny', ROOT / 'models'), gpu_id=args.gpu_id)
        except Exception as exc:
            data['test_inferencji'] = {'ok': False, 'error': friendly_error(exc)}
    print(json.dumps(data, indent=2, ensure_ascii=False))
    if args.test and not data.get('test_inferencji', {}).get('ok'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
