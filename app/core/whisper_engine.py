from __future__ import annotations

import gc
import logging
import threading

from app.core import audio_processor
from app.core.gpu_manager import check_cuda_libraries, configure_cuda_dlls, friendly_error, gpu_stats, resolve_cuda_device
from app.core.speaker_diarization import SpeakerCancelled, label_segments


class Cancelled(Exception):
    """The current job was cancelled between inference steps."""


class WhisperEngine:
    def __init__(self, models):
        self.models = models
        self.model = None
        self.key = None
        self.loaded_name = None
        self.loaded_gpu = None
        self.guard = threading.RLock()

    def unload(self):
        with self.guard:
            model = self.model
            self.model = None
            self.key = None
            self.loaded_name = None
            self.loaded_gpu = None
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

        selected_gpu = None
        if options.device == 'cuda':
            check_cuda_libraries()
            selected_gpu = resolve_cuda_device(options.gpu_id)
            device_index = selected_gpu['index']
            supported = ctranslate2.get_supported_compute_types('cuda', device_index)
            if options.compute_type == 'auto':
                stats = gpu_stats(selected_gpu['id'])
                total, used = stats.get('memory_total'), stats.get('memory_used')
                free = total - used if total is not None and used is not None else 0
                recommended = {'tiny': 1.0, 'base': 1.0, 'small': 1.5, 'medium': 3.5,
                               'large-v3': 7.0, 'turbo': 4.0}[options.model] * 1024**3
                preferred = 'int8_float16' if free > 0 and free < recommended else 'float16'
                precision = next((item for item in (preferred, 'float16', 'int8_float16', 'float32', 'int8_float32')
                                  if item in supported), None)
                if precision is None:
                    raise RuntimeError('Wybrana karta GPU nie obsługuje żadnej dostępnej precyzji CTranslate2.')
            else:
                precision = options.compute_type
            if precision not in supported:
                raise RuntimeError(f'Precyzja {precision} nie jest obsługiwana na tej karcie. Dostępne: {sorted(supported)}')
        else:
            device_index = 0
            if options.compute_type != 'auto':
                raise ValueError('Dla CPU wybierz precyzję automatyczną; float16 i int8_float16 są ustawieniami GPU.')
            precision = 'int8'
        path = self.models.path(options.model, options.cache_dir)
        key = (str(path), options.device, precision,
               (selected_gpu['id'], device_index) if selected_gpu else None)
        if key != self.key:
            self.unload()
            self.model = WhisperModel(str(path), device=options.device, device_index=device_index,
                                      compute_type=precision, local_files_only=True)
            if self.model.model.device != options.device:
                self.unload()
                raise RuntimeError('Silnik uruchomił się na innym urządzeniu niż wybrane.')
            if selected_gpu:
                actual_indexes = self.model.model.device_index
                actual_indexes = [actual_indexes] if isinstance(actual_indexes, int) else list(actual_indexes)
                if actual_indexes != [device_index]:
                    self.unload()
                    raise RuntimeError('Silnik uruchomił się na innej karcie GPU niż wybrana.')
            actual_precision = self.model.model.compute_type
            if actual_precision != precision:
                self.unload()
                raise RuntimeError(f'Silnik nie zastosował wybranej precyzji {precision}; użył {actual_precision}.')
            self.key = key
            self.loaded_name = options.model
            self.loaded_gpu = selected_gpu
        return self.model, precision

    def transcribe(self, path, options, on_progress, cancelled, on_segment=None, on_started=None, speaker_tracker=None):
        return self._transcribe_chunks(audio_processor.chunks(path), options, on_progress, cancelled,
                                       on_segment=on_segment, on_started=on_started, speaker_tracker=speaker_tracker)

    def transcribe_samples(self, samples, options, cancelled):
        """Transcribe one in-memory, mono 16 kHz audio window."""
        return self._transcribe_chunks([(0.0, samples)], options, lambda _: None, cancelled)

    def prepare(self, options):
        if options.task == 'translate' and options.model == 'turbo':
            raise ValueError('Model turbo nie jest przeznaczony do tłumaczenia. Wybierz np. large-v3.')
        with self.guard:
            self._model(options)

    def _transcribe_chunks(self, chunks, options, on_progress, cancelled, on_segment=None, on_started=None, speaker_tracker=None):
        if options.task == 'translate' and options.model == 'turbo':
            raise ValueError('Model turbo nie jest przeznaczony do tłumaczenia. Wybierz np. large-v3.')
        with self.guard:
            try:
                model, precision = self._model(options)
                if on_started:
                    on_started()
                result = {'segments': [], 'language': None, 'device': options.device,
                          'model': options.model, 'compute_type': precision}
                if speaker_tracker:
                    result['diarization'] = True
                if self.loaded_gpu:
                    result.update(gpu_id=self.loaded_gpu['id'], gpu_index=self.loaded_gpu['index'],
                                  gpu_name=self.loaded_gpu['name'])
                language = options.language or None
                has_audio = False
                for offset, samples in chunks:
                    has_audio = True
                    if cancelled():
                        raise Cancelled()
                    segments, info = model.transcribe(samples, language=language, task=options.task,
                        beam_size=options.beam_size, temperature=options.temperature,
                        best_of=options.best_of, patience=options.patience,
                        repetition_penalty=options.repetition_penalty,
                        no_repeat_ngram_size=options.no_repeat_ngram_size,
                        no_speech_threshold=options.no_speech_threshold,
                        condition_on_previous_text=options.condition_on_previous_text,
                        vad_filter=options.vad, word_timestamps=bool(options.word_timestamps or speaker_tracker),
                        vad_parameters={'threshold': options.vad_threshold,
                                        'min_silence_duration_ms': options.vad_min_silence_ms,
                                        'speech_pad_ms': options.vad_speech_pad_ms} if options.vad else None,
                        initial_prompt=options.initial_prompt or None,
                        hotwords=options.hotwords or None)
                    if not result['language']:
                        result['language'] = info.language
                        result['language_probability'] = info.language_probability
                        if not language:
                            language = info.language
                    pending = []
                    for segment in segments:
                        if cancelled():
                            raise Cancelled()
                        item = {'start': offset + segment.start, 'end': offset + segment.end, 'text': segment.text.strip()}
                        if segment.words:
                            item['words'] = [{'start': offset + w.start, 'end': offset + w.end,
                                              'word': w.word, 'probability': w.probability} for w in segment.words]
                        if speaker_tracker:
                            pending.append(item)
                        else:
                            result['segments'].append(item)
                            if on_segment:
                                on_segment(item)
                            on_progress(item['end'])
                    if speaker_tracker and pending:
                        turns = speaker_tracker.turns(samples, cancelled)
                        for item in label_segments(pending, turns, offset, keep_words=options.word_timestamps):
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
            except SpeakerCancelled:
                raise Cancelled() from None
            except Cancelled:
                raise
            except Exception as exc:
                raise RuntimeError(friendly_error(exc)) from exc
            finally:
                if not options.keep_model:
                    self.unload()
