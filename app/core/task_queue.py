from __future__ import annotations

import logging
import threading
import time

from app.config import Settings
from app.core.audio_processor import probe
from app.core.exporter import export_all
from app.core.model_manager import DownloadCancelled
from app.core.speaker_diarization import SpeakerCancelled, SpeakerDiarizer
from app.core.whisper_engine import Cancelled, WhisperEngine
from app.core.vulkan_engine import VulkanEngine


def estimate_eta(processed, duration, elapsed):
    """Estimate remaining inference time from completed audio and measured wall time."""
    if duration is None or duration <= 0 or processed < 5 or elapsed < .5:
        return None
    if processed >= duration:
        return 0.0
    return max(0.0, elapsed * (duration - processed) / processed)


class TaskQueue:
    def __init__(self, db, models):
        self.db, self.models = db, models
        self.gpu_engine = WhisperEngine(models)
        self.vulkan_engine = VulkanEngine(models)
        self.stop_event = threading.Event()
        self.threads = []
        self.claim_lock = threading.Lock()
        self.gpu_slot = threading.Lock()
        self._timing_lock = threading.Lock()
        self._started_jobs = {}
        self.running = 0

    def elapsed(self, job):
        """Return a running task's elapsed time, or its saved final duration."""
        if job['status'] == 'waiting':
            return None
        with self._timing_lock:
            started = self._started_jobs.get(job['id'])
        if started is not None and job['status'] not in ('done', 'error', 'cancelled'):
            return max(0.0, time.monotonic() - started)
        return job.get('elapsed')

    def start(self):
        if self.threads:
            return
        for index in range(2):
            worker = threading.Thread(target=self._worker, args=(index,), name=f'whysper-worker-{index}', daemon=True)
            self.threads.append(worker)
            worker.start()

    def stop(self, *, unload_gpu=True):
        self.stop_event.set()
        for worker in self.threads:
            worker.join(timeout=5)
        if unload_gpu and not any(worker.is_alive() for worker in self.threads):
            self.gpu_engine.unload()
            self.vulkan_engine.unload()

    def _worker(self, index):
        cpu_engine = WhisperEngine(self.models)
        while not self.stop_event.is_set():
            with self.claim_lock:
                if self.running < self.db.settings().workers:
                    job = self.db.claim()
                    if job:
                        self.running += 1
                else:
                    job = None
            if not job:
                self.stop_event.wait(0.5)
                continue
            try:
                self._run(job, cpu_engine)
            finally:
                with self.claim_lock:
                    self.running -= 1
        cpu_engine.unload()

    def _run(self, job, cpu_engine):
        started = time.monotonic()
        with self._timing_lock:
            self._started_jobs[job['id']] = started
        try:
            self._run_started(job, cpu_engine, started)
        finally:
            with self._timing_lock:
                self._started_jobs.pop(job['id'], None)

    def _run_started(self, job, cpu_engine, started):
        job_id = job['id']
        options = Settings.model_validate_json(job['options'])

        def cancelled():
            current = self.db.get(job_id)
            return self.stop_event.is_set() or not current or bool(current['cancel_requested'])

        try:
            if cancelled():
                raise Cancelled()
            if options.device == 'vulkan':
                self.vulkan_engine.check_prerequisites()
            model_args = (options.model, options.cache_dir, 'vulkan') if options.device == 'vulkan' else (options.model, options.cache_dir)
            if not self.models.available(*model_args):
                self.db.update(job_id, status='downloading', message='Pobieranie modelu')
                self.models.download(*model_args)
            if cancelled():
                raise Cancelled()
            self.db.update(job_id, status='preparing', message='Odczytywanie nagrania')
            duration = probe(job['path'])
            self.db.update(job_id, duration=duration, progress=0 if duration else None)

            speaker_tracker = None
            if options.diarization:
                self.db.update(job_id, message='Przygotowywanie modeli rozmówców (pierwsze użycie: około 46 MB)')
                speaker_tracker = SpeakerDiarizer(options.cache_dir, cancelled)

            inference_started = None
            sequence = 0

            def started_inference():
                nonlocal inference_started
                inference_started = time.monotonic()

            def on_segment(segment):
                nonlocal sequence
                self.db.append_segment(job_id, sequence, segment)
                sequence += 1

            def progress(processed):
                value = min(1.0, max(0, processed / duration)) if duration else None
                current = time.monotonic()
                eta = estimate_eta(processed, duration, current - inference_started) if inference_started else None
                self.db.update(job_id, progress=value, elapsed=current - started, eta_seconds=eta)

            engine = self.vulkan_engine if options.device == 'vulkan' else self.gpu_engine if options.device == 'cuda' else cpu_engine
            acquired = False
            try:
                if options.device in ('cuda', 'vulkan'):
                    self.db.update(job_id, message='Oczekiwanie na dostęp do GPU')
                    while not acquired:
                        if cancelled():
                            raise Cancelled()
                        acquired = self.gpu_slot.acquire(timeout=.2)
                if cancelled():
                    raise Cancelled()
                self.db.update(job_id, status='transcribing', message='')
                extra = {'speaker_tracker': speaker_tracker} if speaker_tracker else {}
                result = engine.transcribe(job['path'], options, progress, cancelled,
                                           on_segment=on_segment, on_started=started_inference, **extra)
            finally:
                if acquired:
                    self.gpu_slot.release()
            if cancelled():
                raise Cancelled()
            elapsed = time.monotonic() - started
            result.update(duration=duration, elapsed=elapsed, rtf=(elapsed / duration if duration else None))
            self.db.update(job_id, status='exporting', language=result['language'], eta_seconds=None)
            export_all(result, options.output_dir, job_id)
            if cancelled():
                raise Cancelled()
            if not self.db.complete(job_id, result, elapsed):
                raise Cancelled()
        except (Cancelled, DownloadCancelled, SpeakerCancelled) as exc:
            status = 'error' if self.stop_event.is_set() else 'cancelled'
            if status == 'error':
                message = 'Przerwano zamykanie aplikacji. Można ponowić.'
            elif isinstance(exc, DownloadCancelled):
                message = 'Pobieranie modelu anulowano. Można ponowić zadanie.'
            else:
                message = 'Anulowano.'
            self.db.update(job_id, status=status, message=message, eta_seconds=None, elapsed=time.monotonic()-started)
        except Exception as exc:
            logging.exception('Błąd zadania %s', job_id)
            self.db.update(job_id, status='error', message=str(exc), eta_seconds=None, elapsed=time.monotonic()-started)
