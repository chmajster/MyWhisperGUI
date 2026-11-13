import logging
from pathlib import Path
import uuid

import av
import numpy as np

from app.config import UPLOADS

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.webm'}
EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.opus'} | VIDEO_EXTENSIONS


def extract_video_audio(path):
    """Write and verify the audio before the caller removes the uploaded video."""
    path = Path(path)
    audio_path = path.with_suffix('.flac')
    sample_rate = 16000
    samples_written = 0
    created = False
    try:
        with audio_path.open('xb') as audio_file:
            created = True
            with av.open(str(path), mode='r') as source, av.open(audio_file, mode='w', format='flac') as output:
                if not source.streams.audio:
                    raise ValueError('Plik nie zawiera ścieżki audio.')
                stream = output.add_stream('flac', rate=sample_rate)
                stream.layout = 'mono'
                resampler = av.AudioResampler(format='s16', layout='mono', rate=sample_rate)

                def write_frame(frame):
                    nonlocal samples_written
                    samples_written += frame.samples
                    frame.pts = None
                    for packet in stream.encode(frame):
                        output.mux(packet)

                for frame in source.decode(audio=0):
                    for converted in resampler.resample(frame):
                        write_frame(converted)
                for converted in resampler.resample(None):
                    write_frame(converted)
                if not samples_written:
                    raise ValueError('Nagranie nie zawiera próbek audio.')
                for packet in stream.encode(None):
                    output.mux(packet)

        # Decode the finished file before deleting its source; this also detects
        # incomplete output and ensures it contains every sample we encoded.
        with av.open(str(audio_path), mode='r') as check:
            samples_read = sum(frame.samples for frame in check.decode(audio=0))
        if samples_read != samples_written:
            raise ValueError('Niekompletny plik audio po konwersji.')
        return audio_path, audio_path.stat().st_size, samples_written / sample_rate
    except Exception:
        if created:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                logging.exception('Nie można usunąć niekompletnego pliku audio: %s', audio_path)
        raise


def store_upload(file, name, max_bytes, directory=UPLOADS):
    extension = Path(name).suffix.lower()
    if extension not in EXTENSIONS:
        raise ValueError('Nieobsługiwany typ pliku audio lub wideo.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (uuid.uuid4().hex + extension)
    count = 0
    created = False
    try:
        with target.open('xb') as out:
            created = True
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
    except Exception:
        if created:
            target.unlink(missing_ok=True)
        raise

    if extension not in VIDEO_EXTENSIONS:
        return target, count, duration, None

    try:
        audio, audio_size, audio_duration = extract_video_audio(target)
    except Exception:
        logging.exception('Nie można przekonwertować wideo na audio: %s', target)
        return target, count, duration, 'Nie udało się przygotować audio. Zachowano lokalną kopię wideo do transkrypcji.'
    try:
        target.unlink()
    except OSError:
        logging.exception('Nie można usunąć lokalnej kopii wideo: %s', target)
        return audio, audio_size, audio_duration, 'Audio jest gotowe, ale nie udało się usunąć lokalnej kopii wideo.'
    return audio, audio_size, audio_duration, None


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
