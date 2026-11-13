"""Run a CUDA inference in a child process, so native DLL failures are diagnosable."""
import argparse
import json

from app.core.gpu_manager import check_cuda_libraries, configure_cuda_dlls, gpu_stats, resolve_cuda_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('audio_path', nargs='?')
    parser.add_argument('--gpu-id', default='')
    args = parser.parse_args()
    configure_cuda_dlls()
    import numpy as np
    from faster_whisper import WhisperModel

    selected = resolve_cuda_device(args.gpu_id)
    check_cuda_libraries()
    before = gpu_stats(selected['id'])
    model = WhisperModel(args.model_dir, device='cuda', device_index=selected['index'], compute_type='float16', local_files_only=True)
    if model.model.device != 'cuda':
        raise RuntimeError(f'Nieoczekiwane urządzenie silnika: {model.model.device}')
    actual_indexes = model.model.device_index
    actual_indexes = [actual_indexes] if isinstance(actual_indexes, int) else list(actual_indexes)
    if actual_indexes != [selected['index']]:
        raise RuntimeError('Test uruchomił się na innej karcie GPU niż wybrana.')
    audio = args.audio_path or np.zeros(16000, dtype=np.float32)
    segments, info = model.transcribe(audio, beam_size=1, vad_filter=False)
    text = ' '.join(segment.text.strip() for segment in segments).strip()
    after = gpu_stats(selected['id'])
    print(json.dumps({'ok': True, 'device': model.model.device, 'compute_type': model.model.compute_type,
                      'language': info.language, 'text': text, 'gpu_name': selected['name'],
                      'gpu_id': selected['id'], 'gpu_index': selected['index'],
                      'vram_before': before.get('memory_used'), 'vram_during': after.get('memory_used')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
