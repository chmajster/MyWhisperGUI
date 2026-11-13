import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.config import DB_PATH, Settings

ACTIVE = ('downloading', 'preparing', 'transcribing', 'exporting')
TERMINAL = ('done', 'error', 'cancelled')


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class Database:
    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.connect() as conn:
            conn.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL,
                    status TEXT NOT NULL, position INTEGER NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, options TEXT NOT NULL, progress REAL,
                    duration REAL, elapsed REAL, eta_seconds REAL, language TEXT, message TEXT DEFAULT '',
                    result TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS jobs_status_position ON jobs(status, position);
                CREATE TABLE IF NOT EXISTS job_segments (
                    job_id TEXT NOT NULL, sequence INTEGER NOT NULL, segment TEXT NOT NULL,
                    PRIMARY KEY (job_id, sequence)
                );
            ''')
            columns = {row['name'] for row in conn.execute('PRAGMA table_info(jobs)')}
            if 'eta_seconds' not in columns:
                conn.execute('ALTER TABLE jobs ADD COLUMN eta_seconds REAL')
            conn.execute("UPDATE jobs SET status='error', message=?, updated_at=? WHERE status IN ('downloading','preparing','transcribing','exporting')", ('Przerwano podczas zamykania programu. Można ponowić.', now()))

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def settings(self):
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key='main'").fetchone()
        return Settings.model_validate_json(row['value']) if row else Settings()

    def save_settings(self, value: Settings):
        with self.lock, self.connect() as conn:
            conn.execute("INSERT INTO settings VALUES ('main', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (value.model_dump_json(),))

    def add_job(self, name: str, path: str, options: Settings):
        job_id = uuid.uuid4().hex
        with self.lock, self.connect() as conn:
            position = conn.execute('SELECT COALESCE(MAX(position), 0)+1 FROM jobs').fetchone()[0]
            conn.execute('INSERT INTO jobs(id,name,path,status,position,created_at,updated_at,options) VALUES (?,?,?,?,?,?,?,?)',
                         (job_id, name, path, 'waiting', position, now(), now(), options.model_dump_json()))
        return job_id

    def get(self, job_id):
        with self.connect() as conn:
            row = conn.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        return dict(row) if row else None

    def list(self, statuses=None):
        with self.connect() as conn:
            if statuses:
                placeholders = ','.join('?' for _ in statuses)
                rows = conn.execute(f'SELECT * FROM jobs WHERE status IN ({placeholders}) ORDER BY position, created_at', tuple(statuses)).fetchall()
            else:
                rows = conn.execute('SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC').fetchall()
        return [dict(row) for row in rows]

    def claim(self):
        with self.lock, self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute("SELECT * FROM jobs WHERE status='waiting' ORDER BY position, created_at LIMIT 1").fetchone()
            if row:
                conn.execute("UPDATE jobs SET status='preparing', updated_at=?, message='' WHERE id=?", (now(), row['id']))
                return dict(row)
        return None

    def update(self, job_id, **values):
        allowed = {'status', 'progress', 'duration', 'elapsed', 'eta_seconds', 'language', 'message', 'result', 'cancel_requested'}
        if not values or not set(values) <= allowed:
            raise ValueError('Nieprawidłowe pola zadania.')
        values['updated_at'] = now()
        with self.lock, self.connect() as conn:
            conn.execute('UPDATE jobs SET ' + ', '.join(f'{key}=?' for key in values) + ' WHERE id=?', (*values.values(), job_id))

    def cancel(self, job_id):
        with self.lock, self.connect() as conn:
            conn.execute("UPDATE jobs SET message=CASE WHEN status='waiting' THEN 'Anulowano.' ELSE 'Anulowanie po zakończeniu bieżącego etapu…' END, status=CASE WHEN status='waiting' THEN 'cancelled' ELSE status END, cancel_requested=1, updated_at=? WHERE id=? AND status NOT IN ('done','error','cancelled')", (now(), job_id))

    def retry(self, job_id):
        with self.lock, self.connect() as conn:
            position = conn.execute('SELECT COALESCE(MAX(position), 0)+1 FROM jobs').fetchone()[0]
            changed = conn.execute("UPDATE jobs SET status='waiting',position=?,progress=NULL,elapsed=NULL,eta_seconds=NULL,message='',result=NULL,cancel_requested=0,updated_at=? WHERE id=? AND status IN ('error','cancelled')", (position, now(), job_id)).rowcount
            if changed:
                conn.execute('DELETE FROM job_segments WHERE job_id=?', (job_id,))

    def move(self, job_id, direction):
        with self.lock, self.connect() as conn:
            rows = conn.execute("SELECT id,position FROM jobs WHERE status='waiting' ORDER BY position,created_at").fetchall()
            ids = [row['id'] for row in rows]
            if job_id not in ids:
                return
            index = ids.index(job_id)
            other = index + direction
            if not 0 <= other < len(rows):
                return
            conn.execute('UPDATE jobs SET position=? WHERE id=?', (rows[other]['position'], job_id))
            conn.execute('UPDATE jobs SET position=? WHERE id=?', (rows[index]['position'], rows[other]['id']))

    def save_result(self, job_id, result):
        self.update(job_id, result=json.dumps(result, ensure_ascii=False))

    def append_segment(self, job_id, sequence, segment):
        with self.lock, self.connect() as conn:
            conn.execute('INSERT INTO job_segments(job_id, sequence, segment) VALUES (?,?,?)',
                         (job_id, sequence, json.dumps(segment, ensure_ascii=False)))

    def live_segments(self, job_id, after=-1):
        with self.connect() as conn:
            rows = conn.execute('SELECT sequence, segment FROM job_segments WHERE job_id=? AND sequence>? ORDER BY sequence',
                                (job_id, after)).fetchall()
        return [{'sequence': row['sequence'], **json.loads(row['segment'])} for row in rows]

    def complete(self, job_id, result, elapsed):
        with self.lock, self.connect() as conn:
            cursor = conn.execute("UPDATE jobs SET status='done', progress=1.0, elapsed=?, eta_seconds=NULL, result=?, message='', updated_at=? WHERE id=? AND status='exporting' AND cancel_requested=0",
                                  (elapsed, json.dumps(result, ensure_ascii=False), now(), job_id))
            return cursor.rowcount == 1

    def edit_segment(self, job_id, index, text):
        with self.lock, self.connect() as conn:
            row = conn.execute("SELECT result FROM jobs WHERE id=? AND status='done'", (job_id,)).fetchone()
            if not row or not row['result']:
                raise ValueError('Nie znaleziono wyniku.')
            result = json.loads(row['result'])
            segment = result['segments'][index]
            if segment['text'] != text:
                segment['text'] = text
                segment.pop('words', None)  # Word timings no longer describe edited text.
                conn.execute('UPDATE jobs SET result=?,updated_at=? WHERE id=?',
                             (json.dumps(result, ensure_ascii=False), now(), job_id))

    def edit_speaker(self, job_id, index, speaker):
        speaker = speaker.strip()
        if len(speaker) > 80:
            raise ValueError('Nazwa rozmówcy może mieć najwyżej 80 znaków.')
        with self.lock, self.connect() as conn:
            row = conn.execute("SELECT result FROM jobs WHERE id=? AND status='done'", (job_id,)).fetchone()
            if not row or not row['result']:
                raise ValueError('Nie znaleziono wyniku.')
            result = json.loads(row['result'])
            result['segments'][index]['speaker'] = speaker or None
            conn.execute('UPDATE jobs SET result=?,updated_at=? WHERE id=?',
                         (json.dumps(result, ensure_ascii=False), now(), job_id))

    def delete(self, job_id):
        with self.lock, self.connect() as conn:
            changed = conn.execute("DELETE FROM jobs WHERE id=? AND status IN ('done','error','cancelled')", (job_id,)).rowcount
            if changed:
                conn.execute('DELETE FROM job_segments WHERE job_id=?', (job_id,))
