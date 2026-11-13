from pathlib import Path
import uuid

import av
import numpy as np

from app.config import UPLOADS

EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.mp4', '.mkv', '.mov', '.avi', '.webm'}


def store_upload(file, name, max_bytes, directory=UPLOADS):
    extension = Path(name).suffix.lower()
    if extension not in EXTENSIONS:
        raise ValueError('Nieobsługiwany typ pliku audio lub wideo.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (uuid.uuid4().hex + extension)
    count = 0
    try:
        with target.open('xb') as out:
            while True:
                part = file.read(1024 * 1024)
                if not part:
                    break
                count += len(part)
                if count > max_bytes:
                    raise ValueError('Plik przekracza dozwolony rozmiar.')
                out.write(part)
        if not count:
            raise ValueError('Plik jest pusty.')
        duration = probe(target)
        return target, count, duration
    except Exception:
        target.unlink(missing_ok=True)
        raise


def probe(path):
    try:
        with av.open(str(path), mode='r') as container:
            if not container.streams.audio:
                raise ValueError('Plik nie zawiera ścieżki audio.')
            stream = container.streams.audio[0]
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            else:
                duration = float(container.duration / av.time_base) if container.duration else 0
            return duration if duration > 0 else None
    except Exception as exc:
        raise ValueError(f'Nie można odczytać pliku audio/wideo: {exc}') from exc


def chunks(path, seconds=30, sample_rate=16000):
    """Decode incrementally; memory usage is bounded by one audio window."""
    limit = sample_rate * seconds
    arrays = []
    count = 0
    offset = 0
    with av.open(str(path), mode='r') as container:
        if not container.streams.audio:
            raise ValueError('Plik nie zawiera ścieżki audio.')
        resampler = av.AudioResampler(format='s16', layout='mono', rate=sample_rate)
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                samples = out.to_ndarray().reshape(-1)
                while samples.size:
                    take = min(limit - count, samples.size)
                    arrays.append(samples[:take].copy())
                    count += take
                    samples = samples[take:]
                    if count == limit:
                        yield offset / sample_rate, np.concatenate(arrays).astype(np.float32) / 32768.0
                        offset += count
                        arrays, count = [], 0
        for out in resampler.resample(None):
            samples = out.to_ndarray().reshape(-1)
            while samples.size:
                take = min(limit - count, samples.size)
                arrays.append(samples[:take].copy())
                count += take
                samples = samples[take:]
                if count == limit:
                    yield offset / sample_rate, np.concatenate(arrays).astype(np.float32) / 32768.0
                    offset += count
                    arrays, count = [], 0
        if count:
            yield offset / sample_rate, np.concatenate(arrays).astype(np.float32) / 32768.0
