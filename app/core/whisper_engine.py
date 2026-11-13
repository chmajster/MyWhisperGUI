from __future__ import annotations

import gc
import logging
import threading

from app.core import audio_processor
from app.core.gpu_manager import check_cuda_libraries, configure_cuda_dlls, friendly_error, gpu_stats


class Cancelled(Exception):
    """The current job was cancelled between inference steps."""


class WhisperEngine:
    def __init__(self, models):
        self.models = models
        self.model = None
        self.key = None
        self.loaded_name = None
        self.guard = threading.RLock()

    def unload(self):
        with self.guard:
            model = self.model
            self.model = None
            self.key = None
            self.loaded_name = None
            if model is not None:
                try:
                    model.model.unload_model(to_cpu=False)
                except Exception:
                    logging.exception('Nie udało się zwolnić modelu CTranslate2')
            del model
            gc.collect()

    def _model(self, options):
        configure_cuda_dlls()
        import ctranslate2
        from faster_whisper import WhisperModel

        if options.device == 'cuda':
            check_cuda_libraries()
            if ctranslate2.get_cuda_device_count() < 1:
                raise RuntimeError('Nie wykryto urządzenia CUDA. Sprawdź sterownik NVIDIA lub świadomie wybierz CPU.')
            supported = ctranslate2.get_supported_compute_types('cuda', 0)
            if options.compute_type == 'auto':
                stats = gpu_stats()
                free = stats.get('memory_total', 0) - stats.get('memory_used', 0)
                recommended = {'tiny': 1.0, 'base': 1.0, 'small': 1.5, 'medium': 3.5,
                               'large-v3': 7.0, 'turbo': 4.0}[options.model] * 1024**3
                precision = 'int8_float16' if free > 0 and free < recommended and 'int8_float16' in supported else 'float16'
            else:
                precision = options.compute_type
            if precision not in supported:
                raise RuntimeError(f'Precyzja {precision} nie jest obsługiwana na tej karcie. Dostępne: {sorted(supported)}')
        else:
            if options.compute_type != 'auto':
                raise ValueError('Dla CPU wybierz precyzję automatyczną; float16 i int8_float16 są ustawieniami GPU.')
            precision = 'int8'
        path = self.models.path(options.model, options.cache_dir)
        key = (str(path), options.device, precision)
        if key != self.key:
            self.unload()
            self.model = WhisperModel(str(path), device=options.device, device_index=0,
                                      compute_type=precision, local_files_only=True)
            if self.model.model.device != options.device:
                self.unload()
                raise RuntimeError('Silnik uruchomił się na innym urządzeniu niż wybrane.')
            actual_precision = self.model.model.compute_type
            if actual_precision != precision:
                self.unload()
                raise RuntimeError(f'Silnik nie zastosował wybranej precyzji {precision}; użył {actual_precision}.')
            self.key = key
            self.loaded_name = options.model
        return self.model, precision

    def transcribe(self, path, options, on_progress, cancelled, on_segment=None, on_started=None):
        if options.task == 'translate' and options.model == 'turbo':
            raise ValueError('Model turbo nie jest przeznaczony do tłumaczenia. Wybierz np. large-v3.')
        with self.guard:
            try:
                model, precision = self._model(options)
                if on_started:
                    on_started()
                result = {'segments': [], 'language': None, 'device': options.device,
                          'model': options.model, 'compute_type': precision}
                language = options.language or None
                has_audio = False
                for offset, samples in audio_processor.chunks(path):
                    has_audio = True
                    if cancelled():
                        raise Cancelled()
                    segments, info = model.transcribe(samples, language=language, task=options.task,
                        beam_size=options.beam_size, temperature=options.temperature,
                        vad_filter=options.vad, word_timestamps=options.word_timestamps,
                        initial_prompt=options.initial_prompt or None)
                    if not result['language']:
                        result['language'] = info.language
                        result['language_probability'] = info.language_probability
                        if not language:
                            language = info.language
                    for segment in segments:
                        if cancelled():
                            raise Cancelled()
                        item = {'start': offset + segment.start, 'end': offset + segment.end, 'text': segment.text.strip()}
                        if segment.words:
                            item['words'] = [{'start': offset + w.start, 'end': offset + w.end,
                                              'word': w.word, 'probability': w.probability} for w in segment.words]
                        result['segments'].append(item)
                        if on_segment:
                            on_segment(item)
                        on_progress(item['end'])
                    # A silent chunk is still fully processed.
                    on_progress(offset + len(samples) / 16000)
                if not has_audio:
                    raise ValueError('Nagranie nie zawiera żadnych próbek audio do przetworzenia.')
                if cancelled():
                    raise Cancelled()
                return result
            except Cancelled:
                raise
            except Exception as exc:
                raise RuntimeError(friendly_error(exc)) from exc
            finally:
                if not options.keep_model:
                    self.unload()
