import csv
import io
import json
import os
import tempfile
from pathlib import Path


def timestamp(seconds, separator='.'):
    ms = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(ms, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, millis = divmod(remainder, 1000)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}'


def render(result, format):
    segments = result.get('segments', [])
    if format == 'txt':
        return '\n'.join(s['text'].strip() for s in segments).strip() + '\n'
    if format in ('srt', 'vtt'):
        separator = ',' if format == 'srt' else '.'
        blocks = []
        for index, segment in enumerate(segments, 1):
            timing = f"{timestamp(segment['start'], separator)} --> {timestamp(segment['end'], separator)}"
            prefix = f'{index}\n' if format == 'srt' else ''
            blocks.append(prefix + timing + '\n' + segment['text'].strip())
        return ('WEBVTT\n\n' if format == 'vtt' else '') + '\n\n'.join(blocks) + '\n'
    if format == 'json':
        return json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if format == 'csv':
        output = io.StringIO(newline='')
        writer = csv.writer(output)
        writer.writerow(['start', 'end', 'text'])
        writer.writerows((s['start'], s['end'], s['text'].strip()) for s in segments)
        return output.getvalue()
    raise ValueError(f'Nieznany format: {format}')


def export_all(result, directory, basename):
    target = Path(directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    # Nazwa pochodzi od identyfikatora zadania, nie od nazwy pliku użytkownika.
    if not basename.isalnum():
        raise ValueError('Nieprawidłowy identyfikator eksportu.')
    for format in ('txt', 'srt', 'vtt', 'json', 'csv'):
        destination = target / f'{basename}.{format}'
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=target, delete=False) as stream:
            temporary = stream.name
            try:
                stream.write(render(result, format))
            except Exception:
                Path(temporary).unlink(missing_ok=True)
                raise
        os.replace(temporary, destination)
