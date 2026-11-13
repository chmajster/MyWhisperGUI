import threading
import uuid
import wave

import numpy as np
import sounddevice as sd

from app.config import RECORDINGS


class Recorder:
    def __init__(self, microphone_lock=None):
        self.lock = threading.RLock()
        self.microphone_lock = microphone_lock if microphone_lock is not None else threading.Lock()
        self.owns_microphone = False
        self.stream = None
        self.output = None
        self.path = None
        self.paused = False
        self.frames = 0
        self.level = 0.0
        self.error = ''
        self.rate = 16000

    @staticmethod
    def devices():
        return {i: f"{d['name']} ({d['max_input_channels']} wejść)"
                for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0}

    def start(self, device=None):
        with self.lock:
            if self.stream:
                raise RuntimeError('Nagrywanie już trwa.')
            if not self.microphone_lock.acquire(blocking=False):
                raise RuntimeError('Mikrofon jest już używany przez inną funkcję aplikacji.')
            self.owns_microphone = True
            try:
                self._start_locked(device)
            except Exception:
                self.owns_microphone = False
                self.microphone_lock.release()
                raise

    def _start_locked(self, device):
        self.rate = 16000
        try:
            sd.check_input_settings(device=device, channels=1, dtype='int16', samplerate=self.rate)
        except sd.PortAudioError:
            self.rate = int(sd.query_devices(device, 'input')['default_samplerate'])
            sd.check_input_settings(device=device, channels=1, dtype='int16', samplerate=self.rate)
        RECORDINGS.mkdir(parents=True, exist_ok=True)
        path = RECORDINGS / (uuid.uuid4().hex + '.wav')
        output = wave.open(str(path), 'wb')
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(self.rate)
        self.path, self.output = path, output
        self.frames, self.level, self.error, self.paused = 0, 0.0, '', False

        def callback(data, frames, timing, status):
            with self.lock:
                if status:
                    self.error = str(status)
                if not self.paused and self.output:
                    self.output.writeframesraw(data.tobytes())
                    self.frames += frames
                    self.level = float(np.max(np.abs(data.astype(np.float32))) / 32768) if data.size else 0
                else:
                    self.level = 0
        try:
            self.stream = sd.InputStream(device=device, samplerate=self.rate, channels=1,
                                         dtype='int16', callback=callback)
            self.stream.start()
        except Exception:
            if self.stream:
                self.stream.close()
            self.stream = None
            self.output.close()
            self.output = None
            path.unlink(missing_ok=True)
            raise

    def pause(self):
        with self.lock:
            if not self.stream:
                raise RuntimeError('Brak aktywnego nagrania.')
            self.paused = not self.paused
            return self.paused

    def stop(self):
        # Do not hold the callback lock while waiting for PortAudio to stop.
        with self.lock:
            stream = self.stream
            self.stream = None
        if not stream:
            raise RuntimeError('Brak aktywnego nagrania.')
        try:
            stream.stop()
        finally:
            try:
                stream.close()
            finally:
                with self.lock:
                    try:
                        self.output.close()
                    finally:
                        self.output = None
                        self.level = 0
                        if self.owns_microphone:
                            self.owns_microphone = False
                            self.microphone_lock.release()
        return self.path

    def status(self):
        with self.lock:
            return {'active': self.stream is not None, 'paused': self.paused, 'seconds': self.frames / self.rate,
                    'level': self.level, 'path': self.path, 'error': self.error}
