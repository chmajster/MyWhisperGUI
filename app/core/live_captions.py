"""Bounded, local microphone capture and chunked live transcription."""
import logging
from pathlib import Path
import queue
import threading

import av
import numpy as np
import sounddevice as sd

from app.core.gpu_manager import friendly_error
from app.core.speaker_diarization import SpeakerCancelled, SpeakerDiarizer, label_segments
from app.core.whisper_engine import Cancelled, WhisperEngine


class MicrophoneError(RuntimeError):
    """An audio input could not be opened."""


def audio_16khz(samples, sample_rate):
    """Convert a mono int16 capture window to Whisper's float32 format."""
    samples = np.asarray(samples, dtype=np.int16).reshape(-1)
    if sample_rate == 16000:
        return samples.astype(np.float32) / 32768.0
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples).reshape(1, -1), format='s16', layout='mono')
    frame.sample_rate = sample_rate
    resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
    frames = [*resampler.resample(frame), *resampler.resample(None)]
    if not frames:
        return np.empty(0, dtype=np.float32)
    return np.concatenate([item.to_ndarray().reshape(-1) for item in frames]).astype(np.float32) / 32768.0


class LiveCaptions:
    MAX_PENDING = 4
    DISPLAY_LINES = 50

    def __init__(self, models, tasks, microphone_lock):
        self.models, self.tasks, self.microphone_lock = models, tasks, microphone_lock
        self.lock = threading.RLock()
        self.phase = 'idle'
        self.owner = None
        self.options = None
        self.error = ''
        self.language = None
        self.lines = []
        self.revision = 0
        self.dropped_seconds = 0.0
        self.input_warnings = 0
        self.level = 0.0
        self._thread = None
        self._stream = None
        self._closing_stream = False
        self._starting_stream = False
        self._capture_closed = threading.Event()
        self._capture_closed.set()
        self._capture_stop = threading.Event()
        self._abort = threading.Event()
        self._pending = queue.Queue(maxsize=self.MAX_PENDING)
        self._buffer = None
        self._filled = 0
        self._rate = 16000

    @staticmethod
    def devices():
        hostapis = sd.query_hostapis()
        return {index: f"{item['name']} · {hostapis[item['hostapi']]['name']}"
                for index, item in enumerate(sd.query_devices()) if item['max_input_channels'] > 0}

    @staticmethod
    def default_device():
        try:
            index = int(sd.default.device[0])
            return index if index >= 0 else None
        except (TypeError, ValueError, IndexError):
            return None

    def start(self, device, options, chunk_seconds, owner):
        if not 2 <= chunk_seconds <= 15:
            raise ValueError('Długość porcji musi wynosić od 2 do 15 sekund.')
        if options.task != 'transcribe':
            raise ValueError('Napisy na żywo obsługują obecnie transkrypcję w języku mowy.')
        if self.models.is_downloading(options.model, options.cache_dir):
            raise ValueError('Poczekaj na zakończenie pobierania wybranego modelu.')
        if not self.models.available(options.model, options.cache_dir):
            raise ValueError('Najpierw pobierz wybrany model w zakładce Modele AI.')
        with self.lock:
            if (self._thread and self._thread.is_alive()) or not self._capture_closed.is_set():
                raise RuntimeError('Napisy na żywo są już uruchomione lub są zatrzymywane.')
            self.owner, self.options = owner, options.model_copy(deep=True)
            self.phase, self.error, self.language = 'preparing', '', None
            self.lines.clear()
            self.revision += 1
            self.dropped_seconds, self.input_warnings, self.level = 0.0, 0, 0.0
            self._capture_stop, self._abort = threading.Event(), threading.Event()
            self._pending = queue.Queue(maxsize=self.MAX_PENDING)
            self._buffer, self._filled = None, 0
            self._thread = threading.Thread(target=self._work, args=(device, int(chunk_seconds)),
                                            name='whysper-live-captions', daemon=True)
            self._thread.start()

    def _put_locked(self, samples):
        try:
            self._pending.put_nowait(samples)
        except queue.Full:
            try:
                discarded = self._pending.get_nowait()
                self.dropped_seconds += len(discarded) / self._rate
            except queue.Empty:
                pass
            self._pending.put_nowait(samples)

    def _capture_callback(self, data, frames, timing, status):
        try:
            with self.lock:
                if self._capture_stop.is_set() or self._buffer is None:
                    return
                if status:
                    self.input_warnings += 1
                samples = np.asarray(data, dtype=np.int16).reshape(-1)
                self.level = float(np.max(np.abs(samples.astype(np.float32))) / 32768) if samples.size else 0.0
                while samples.size:
                    count = min(len(self._buffer) - self._filled, samples.size)
                    self._buffer[self._filled:self._filled + count] = samples[:count]
                    self._filled += count
                    samples = samples[count:]
                    if self._filled == len(self._buffer):
                        self._put_locked(self._buffer.copy())
                        self._filled = 0
        except Exception:
            logging.exception('Błąd przechwytywania dźwięku do napisów')
            with self.lock:
                self.error = 'Nie udało się odebrać dźwięku z mikrofonu.'
                self._capture_stop.set()
                self._abort.set()

    def _open_capture(self, device, chunk_seconds):
        if not self.microphone_lock.acquire(blocking=False):
            raise RuntimeError('Mikrofon jest już używany przez inną funkcję aplikacji.')
        stream = None
        registered = False
        try:
            rate = 16000
            try:
                sd.check_input_settings(device=device, channels=1, dtype='int16', samplerate=rate)
            except sd.PortAudioError:
                rate = int(sd.query_devices(device, 'input')['default_samplerate'])
                sd.check_input_settings(device=device, channels=1, dtype='int16', samplerate=rate)
            stream = sd.InputStream(device=device, samplerate=rate, channels=1,
                                    dtype='int16', callback=self._capture_callback)
            with self.lock:
                self._rate = rate
                self._buffer = np.empty(rate * chunk_seconds, dtype=np.int16)
                self._filled = 0
                self._stream = stream
                self._closing_stream = False
                self._capture_closed.clear()
                registered = True
                self._starting_stream = not self._capture_stop.is_set()
            if self._starting_stream:
                try:
                    stream.start()
                    with self.lock:
                        if not self._capture_stop.is_set():
                            self.phase = 'listening'
                finally:
                    with self.lock:
                        self._starting_stream = False
            if self._capture_stop.is_set():
                self._close_capture()
        except Exception:
            if registered:
                self._close_capture()
            else:
                if stream:
                    stream.close()
                self.microphone_lock.release()
            raise

    def _close_capture(self):
        with self.lock:
            if self._stream is None or self._closing_stream or self._starting_stream:
                return
            stream = self._stream
            self._closing_stream = True

        def close():
            try:
                try:
                    stream.abort()
                finally:
                    stream.close()
            except Exception:
                logging.exception('Nie udało się zamknąć mikrofonu napisów')
                with self.lock:
                    self.error = self.error or 'Nie udało się prawidłowo zamknąć mikrofonu.'
            finally:
                with self.lock:
                    self._stream = None
                    self._closing_stream = False
                    self.level = 0.0
                self.microphone_lock.release()
                self._capture_closed.set()

        threading.Thread(target=close, name='whysper-live-close', daemon=True).start()

    def stop(self, owner=None, *, finish=True):
        with self.lock:
            if owner is not None and owner != self.owner:
                return False
            if self.phase not in ('preparing', 'preparing_speakers', 'waiting_gpu', 'listening', 'stopping'):
                return False
            if not self._capture_stop.is_set():
                self._capture_stop.set()
                if finish and self._buffer is not None and self._filled:
                    self._put_locked(self._buffer[:self._filled].copy())
                    self._filled = 0
            if not finish:
                self._abort.set()
            self.phase = 'stopping'
        self._close_capture()
        return True

    def _work(self, device, chunk_seconds):
        engine = None
        gpu_acquired = False
        options = self.options
        live_options = options.model_copy(update={'keep_model': True, 'word_timestamps': options.diarization})
        try:
            tracker = None
            if options.diarization:
                with self.lock:
                    self.phase = 'preparing_speakers'
                tracker = SpeakerDiarizer(options.cache_dir,
                                          lambda: self._capture_stop.is_set() or self.tasks.stop_event.is_set())
            if self._capture_stop.is_set() or self.tasks.stop_event.is_set():
                return
            if options.device == 'cuda':
                with self.lock:
                    self.phase = 'waiting_gpu'
                while not self._capture_stop.is_set() and not self.tasks.stop_event.is_set():
                    if self.tasks.gpu_slot.acquire(timeout=.2):
                        gpu_acquired = True
                        break
                if not gpu_acquired:
                    return
                engine = self.tasks.gpu_engine
            else:
                engine = WhisperEngine(self.models)
            if self._capture_stop.is_set() or self.tasks.stop_event.is_set():
                return
            with self.lock:
                self.phase = 'preparing'
            engine.prepare(live_options)
            if self._capture_stop.is_set() or self.tasks.stop_event.is_set():
                return
            try:
                self._open_capture(device, chunk_seconds)
            except Exception as exc:
                raise MicrophoneError(f'Nie udało się uruchomić mikrofonu: {exc}') from exc
            while not self._abort.is_set() and not self.tasks.stop_event.is_set():
                if self._capture_stop.is_set() and self._pending.empty():
                    break
                with self.lock:
                    stream = self._stream
                try:
                    if stream is not None and not self._capture_stop.is_set() and not getattr(stream, 'active', True):
                        raise MicrophoneError('Strumień mikrofonu został przerwany. Sprawdź urządzenie i uruchom napisy ponownie.')
                except sd.PortAudioError as exc:
                    raise MicrophoneError(f'Nie można odczytać stanu mikrofonu: {exc}') from exc
                try:
                    samples = self._pending.get(timeout=.2)
                except queue.Empty:
                    continue
                audio = audio_16khz(samples, self._rate)
                if not audio.size:
                    continue
                # Automatic mode detects each live window separately so a
                # later speaker may use a different language.
                language = options.language or ''
                chunk_options = live_options.model_copy(update={'language': language})
                result = engine.transcribe_samples(audio, chunk_options,
                                                   lambda: self._abort.is_set() or self.tasks.stop_event.is_set())
                if tracker and result['segments']:
                    turns = tracker.turns(audio, lambda: self._abort.is_set() or self.tasks.stop_event.is_set())
                    segments = label_segments(result['segments'], turns)
                    lines = [f"{segment.get('speaker') or 'Nieustalony'}: {segment['text'].strip()}"
                             for segment in segments if segment['text'].strip()]
                else:
                    text = ' '.join(segment['text'].strip() for segment in result['segments'] if segment['text'].strip())
                    lines = [text] if text else []
                if lines:
                    with self.lock:
                        self.lines.extend(lines)
                        self.language = result.get('language') or self.language
                        self.revision += 1
        except (Cancelled, SpeakerCancelled):
            pass
        except Exception as exc:
            logging.exception('Błąd napisów na żywo')
            with self.lock:
                self.error = str(exc) if isinstance(exc, MicrophoneError) else friendly_error(exc)
                self.phase = 'stopping'
        finally:
            self._capture_stop.set()
            self._close_capture()
            if not self._capture_closed.wait(timeout=5):
                with self.lock:
                    self.error = self.error or 'Nie udało się jeszcze zamknąć mikrofonu.'
            try:
                if engine and (options.device != 'cuda' or not options.keep_model):
                    engine.unload()
            except Exception:
                logging.exception('Nie udało się zwolnić modelu napisów na żywo')
                with self.lock:
                    self.error = self.error or 'Nie udało się zwolnić modelu napisów na żywo.'
            finally:
                if gpu_acquired:
                    self.tasks.gpu_slot.release()
            with self.lock:
                if self.error:
                    self.phase = 'error'
                else:
                    self.phase = 'stopped'

    def snapshot(self):
        with self.lock:
            return {'phase': self.phase, 'owner': self.owner, 'error': self.error,
                    'text': '\n'.join(self.lines[-self.DISPLAY_LINES:]), 'line_count': len(self.lines),
                    'revision': self.revision,
                    'language': self.language, 'level': self.level,
                    'dropped_seconds': self.dropped_seconds, 'input_warnings': self.input_warnings,
                    'pending': self._pending.qsize(), 'chunk_seconds': len(self._buffer) / self._rate if self._buffer is not None else None}

    def transcript(self):
        with self.lock:
            return '\n'.join(self.lines)

    def using_model(self, name, cache_dir):
        with self.lock:
            return bool(self._thread and self._thread.is_alive() and self.options
                        and self.options.model == name
                        and Path(self.options.cache_dir).expanduser().resolve() == Path(cache_dir).expanduser().resolve())

    def shutdown(self, timeout=5):
        self.stop(finish=False)
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        return not thread or not thread.is_alive()
