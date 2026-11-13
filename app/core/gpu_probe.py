"""Run a CUDA inference in a child process, so native DLL failures are diagnosable."""
import json
import sys

from app.core.gpu_manager import check_cuda_libraries, configure_cuda_dlls, gpu_stats


def main():
    configure_cuda_dlls()
    import ctranslate2
    import numpy as np
    from faster_whisper import WhisperModel

    if not ctranslate2.get_cuda_device_count():
        raise RuntimeError('CTranslate2 nie wykrywa żadnego urządzenia CUDA.')
    check_cuda_libraries()
    before = gpu_stats()
    model = WhisperModel(sys.argv[1], device='cuda', device_index=0, compute_type='float16', local_files_only=True)
    if model.model.device != 'cuda':
        raise RuntimeError(f'Nieoczekiwane urządzenie silnika: {model.model.device}')
    audio = sys.argv[2] if len(sys.argv) > 2 else np.zeros(16000, dtype=np.float32)
    segments, info = model.transcribe(audio, beam_size=1, vad_filter=False)
    text = ' '.join(segment.text.strip() for segment in segments).strip()
    after = gpu_stats()
    print(json.dumps({'ok': True, 'device': model.model.device, 'compute_type': model.model.compute_type,
                      'language': info.language, 'text': text, 'gpu_name': after.get('name'),
                      'vram_before': before.get('memory_used'), 'vram_during': after.get('memory_used')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
