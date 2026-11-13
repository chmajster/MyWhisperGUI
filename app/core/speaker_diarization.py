"""Local speaker diarization for bounded audio windows and live sessions."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import threading
from urllib.request import urlopen

import numpy as np


class SpeakerCancelled(Exception):
    """Speaker preparation or analysis was cancelled."""


_DOWNLOAD_LOCK = threading.Lock()
_MODELS = (
    ('segmentation.onnx',
     'https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx',
     '220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079'),
    ('embedding.onnx',
     'https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx',
     '1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b'),
)


def _check_cancelled(cancelled):
    if cancelled():
        raise SpeakerCancelled()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_models(cache_dir, cancelled=lambda: False):
    """Download fixed, checksum-verified model files only when the feature is used."""
    directory = Path(cache_dir).expanduser().resolve() / 'speaker-diarization'
    while not _DOWNLOAD_LOCK.acquire(timeout=.2):
        _check_cancelled(cancelled)
    try:
        _check_cancelled(cancelled)
        directory.mkdir(parents=True, exist_ok=True)
        for name, url, expected in _MODELS:
            path = directory / name
            if path.is_file() and _sha256(path) == expected:
                continue
            _check_cancelled(cancelled)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(prefix=name + '.', suffix='.part', dir=directory, delete=False) as target:
                    temporary = Path(target.name)
                    digest = hashlib.sha256()
                    with urlopen(url, timeout=30) as response:
                        while True:
                            _check_cancelled(cancelled)
                            block = response.read(256 * 1024)
                            if not block:
                                break
                            target.write(block)
                            digest.update(block)
                _check_cancelled(cancelled)
                if digest.hexdigest() != expected:
                    raise RuntimeError('Pobrany model rozmówców nie przeszedł kontroli integralności.')
                os.replace(temporary, path)
            except SpeakerCancelled:
                raise
            except Exception as exc:
                raise RuntimeError(f'Nie można przygotować modelu rozmówców: {exc}') from exc
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return directory / 'segmentation.onnx', directory / 'embedding.onnx'
    finally:
        _DOWNLOAD_LOCK.release()


def _overlap(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def _unique_samples(samples, turns, speaker, sample_rate=16000):
    """Prefer non-overlapping speech when calculating a voice embedding."""
    pieces = []
    for turn in turns:
        if turn['local'] != speaker:
            continue
        intervals = [(turn['start'], turn['end'])]
        for other in turns:
            if other['local'] == speaker:
                continue
            cut_start, cut_end = other['start'], other['end']
            next_intervals = []
            for start, end in intervals:
                if cut_end <= start or cut_start >= end:
                    next_intervals.append((start, end))
                else:
                    if start < cut_start:
                        next_intervals.append((start, cut_start))
                    if cut_end < end:
                        next_intervals.append((cut_end, end))
            intervals = next_intervals
        for start, end in intervals:
            left = max(0, int(start * sample_rate))
            right = min(len(samples), int(end * sample_rate))
            if right > left:
                pieces.append(samples[left:right])
    return np.concatenate(pieces)[:sample_rate * 20] if pieces else np.empty(0, dtype=np.float32)


class SpeakerDiarizer:
    """Recognize local speaker turns and keep anonymous labels within a session."""
    MIN_EMBEDDING_SECONDS = .6
    MATCH_THRESHOLD = .55
    MAX_SPEAKERS = 20

    def __init__(self, cache_dir, cancelled=lambda: False):
        try:
            import sherpa_onnx as sherpa
        except ImportError as exc:
            raise RuntimeError('Brak silnika rozmówców. Zainstaluj zależności aplikacji zgodnie z README.') from exc
        segmentation, embedding = prepare_models(cache_dir, cancelled)
        _check_cancelled(cancelled)
        embedding_config = sherpa.SpeakerEmbeddingExtractorConfig(model=str(embedding), num_threads=1, provider='cpu')
        config = sherpa.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(segmentation)),
                num_threads=1, provider='cpu'),
            embedding=embedding_config,
            clustering=sherpa.FastClusteringConfig(num_clusters=-1, threshold=.5),
            min_duration_on=.3, min_duration_off=.5)
        if not config.validate():
            raise RuntimeError('Nie można załadować modeli rozmówców.')
        self.diarizer = sherpa.OfflineSpeakerDiarization(config)
        self.extractor = sherpa.SpeakerEmbeddingExtractor(embedding_config)
        self.profiles = []

    def _embedding(self, audio):
        if len(audio) < int(self.MIN_EMBEDDING_SECONDS * 16000):
            return None
        stream = self.extractor.create_stream()
        stream.accept_waveform(16000, np.ascontiguousarray(audio, dtype=np.float32))
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            return None
        vector = np.asarray(self.extractor.compute(stream), dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if np.isfinite(norm) and norm > 0 else None

    def _match(self, vectors):
        # Match all local speakers before updating profiles so two different
        # speakers in one window cannot be assigned the same global label.
        assigned, used = {}, set()
        candidates = sorted(((float(np.dot(vector, profile)), local, index)
                             for local, vector in vectors.items() if vector is not None
                             for index, profile in enumerate(self.profiles)), reverse=True)
        for score, local, index in candidates:
            if score < self.MATCH_THRESHOLD:
                break
            if local not in assigned and index not in used:
                assigned[local] = index
                used.add(index)
        for local, vector in vectors.items():
            if vector is None:
                continue
            if local not in assigned:
                if len(self.profiles) >= self.MAX_SPEAKERS:
                    continue
                assigned[local] = len(self.profiles)
                self.profiles.append(vector)
            else:
                index = assigned[local]
                updated = self.profiles[index] * .8 + vector * .2
                self.profiles[index] = updated / np.linalg.norm(updated)
        return {local: f'Rozmówca {index + 1}' for local, index in assigned.items()}

    def turns(self, samples, cancelled=lambda: False):
        _check_cancelled(cancelled)
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        duration = len(samples) / 16000
        if duration <= 0:
            return []
        # The segmentation network consumes ten-second windows. Pad short live
        # chunks, then clip its output back to the actual captured duration.
        padded = np.pad(samples, (0, max(0, 160000 - len(samples))))
        raw = self.diarizer.process(padded).sort_by_start_time()
        _check_cancelled(cancelled)
        turns = [{'start': max(0., float(item.start)), 'end': min(duration, float(item.end)),
                  'local': int(item.speaker)} for item in raw
                 if float(item.start) < duration and float(item.end) > 0]
        turns = [item for item in turns if item['end'] > item['start']]
        vectors = {}
        for local in dict.fromkeys(item['local'] for item in turns):
            _check_cancelled(cancelled)
            vectors[local] = self._embedding(_unique_samples(samples, turns, local))
        labels = self._match(vectors)
        return [{'start': item['start'], 'end': item['end'], 'speaker': labels.get(item['local'])}
                for item in turns]


def label_segments(segments, turns, offset=0.0, keep_words=False):
    """Assign words to overlapping turns; leave uncertain speech unlabelled."""
    def speaker_for(start, end):
        scores = {}
        for turn in turns:
            speaker = turn.get('speaker')
            if speaker:
                amount = _overlap(start - offset, end - offset, turn['start'], turn['end'])
                scores[speaker] = scores.get(speaker, 0.0) + amount
        ranking = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        if not ranking or ranking[0][1] <= 0:
            return None
        if len(ranking) > 1 and ranking[1][1] >= ranking[0][1] * .8:
            return None
        return ranking[0][0]

    labelled = []
    for segment in segments:
        words = segment.get('words') or []
        # Some backends may return incomplete word timings. Preserve the full
        # transcription instead of reconstructing it from missing words.
        if not words or ''.join(word['word'] for word in words).strip() != segment['text'].strip():
            item = dict(segment)
            if not keep_words:
                item.pop('words', None)
            item['speaker'] = speaker_for(item['start'], item['end'])
            labelled.append(item)
            continue
        groups = []
        for word in words:
            speaker = speaker_for(word['start'], word['end'])
            if not groups or groups[-1][0] != speaker:
                groups.append((speaker, []))
            groups[-1][1].append(word)
        for speaker, group in groups:
            text = ''.join(word['word'] for word in group).strip()
            if not text:
                continue
            item = {'start': group[0]['start'], 'end': group[-1]['end'], 'text': text, 'speaker': speaker}
            if keep_words:
                item['words'] = group
            labelled.append(item)
    return labelled
