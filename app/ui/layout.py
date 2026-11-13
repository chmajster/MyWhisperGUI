from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import psutil
from nicegui import run, ui

from app.config import MODEL_REPOS, Settings
from app.core.audio_processor import store_upload
from app.core.exporter import export_all, timestamp
from app.core.gpu_manager import diagnostics, gpu_stats, inference_test

STATUS = {'waiting': 'Oczekuje', 'downloading': 'Pobieranie modelu', 'preparing': 'Przygotowanie',
          'transcribing': 'Transkrypcja', 'exporting': 'Eksport', 'done': 'Ukończono',
          'error': 'Błąd', 'cancelled': 'Anulowano'}
NAV = [('dashboard', 'Dashboard', 'space_dashboard'), ('transcription', 'Transkrypcja', 'graphic_eq'),
       ('recording', 'Nagrywanie', 'mic_none'), ('queue', 'Kolejka', 'view_list'),
       ('history', 'Historia', 'history'), ('models', 'Modele AI', 'memory'),
       ('settings', 'Ustawienia', 'tune'), ('diagnostics', 'Diagnostyka', 'monitor_heart')]

CSS = '''
<style>
:root { --accent:#a78bfa; --line:rgba(148,163,184,.16); }
body { font-family:Inter,Segoe UI,sans-serif; }
.body--dark { background:#0b0e16; color:#edf1fa; }
.body--light { background:#f3f5fa; color:#1e293b; }
.shell { width:100%; min-height:100vh; gap:0; }
.sidebar { width:225px; min-height:100vh; background:#111522; border-right:1px solid var(--line); padding:28px 16px; }
.body--light .sidebar { background:#fff; }
.brand-icon { background:#a78bfa; color:#111321; border-radius:11px; padding:8px; }
.brand-title { font-size:19px; font-weight:750; letter-spacing:-.04em; }
.eyebrow { font-size:10px; letter-spacing:.15em; font-weight:700; color:#79849b; }
.nav-btn { width:100%; justify-content:flex-start; border-radius:9px; min-height:43px; color:#a8b2c7; text-transform:none; font-weight:500; }
.nav-btn.active { background:rgba(167,139,250,.13); color:#c4b5fd; }
.main { flex:1; min-width:0; padding:36px 46px 65px; max-width:1600px; }
.page-title { font-size:28px; letter-spacing:-.045em; font-weight:750; line-height:1.2; }
.muted { color:#8994aa; }
.panel { border:1px solid var(--line); background:#151b2a; border-radius:15px; box-shadow:none; padding:22px; }
.body--light .panel { background:white; }
.metric { min-width:190px; flex:1; min-height:132px; }
.metric-value { font-size:25px; font-weight:700; letter-spacing:-.04em; }
.section-title { font-size:17px; font-weight:700; letter-spacing:-.02em; }
.hint { font-size:12px; color:#8994aa; }
.status-chip { background:rgba(167,139,250,.14); color:#c4b5fd; border-radius:6px; padding:3px 9px; font-size:12px; }
.model-row { border:1px solid var(--line); border-radius:12px; padding:16px; margin-top:10px; }
.model-row.ready { border-color:rgba(52,211,153,.45); background:rgba(16,185,129,.06); }
.model-badge { padding:5px 10px; border-radius:7px; font-size:11px; font-weight:700; letter-spacing:.04em; background:rgba(148,163,184,.12); color:#a8b2c7; }
.model-badge.ready { background:rgba(16,185,129,.16); color:#6ee7b7; }
.body--light .model-badge.ready { color:#047857; }
.live-feed { background:rgba(2,6,23,.24); border:1px solid var(--line); border-radius:10px; }
.body--light .live-feed { background:#f8fafc; }
.q-field--outlined .q-field__control { border-radius:9px; }
.q-btn { letter-spacing:0; }
.job-row { border-top:1px solid var(--line); padding:16px 0; }
.record-dot { color:#fb7185; font-size:30px; }
@media(max-width:800px) { .sidebar { width:70px; padding:18px 8px; } .sidebar .q-btn__content span.block, .brand-title, .brand-sub, .side-caption {display:none;} .nav-btn {justify-content:center;} .main {padding:25px 18px;} .model-row, .queue-job-header {flex-wrap:wrap !important;} .model-row > .grow, .queue-job-header > .grow {min-width:100%;} }
</style>
'''


def human_size(size):
    if size is None:
        return '—'
    value = float(size)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if value < 1024 or unit == 'TB':
            return f'{value:.1f} {unit}'
        value /= 1024


def short_time(seconds):
    if seconds is None:
        return '—'
    return timestamp(seconds).split('.')[0]


def eta_text(job):
    if job['status'] != 'transcribing':
        return 'ETA: dostępne po rozpoczęciu transkrypcji' if job['status'] in ('waiting', 'downloading', 'preparing') else ''
    eta = job.get('eta_seconds')
    if eta is None:
        return 'ETA: obliczanie na podstawie tempa transkrypcji…'
    try:
        updated = datetime.fromisoformat(job['updated_at'])
        age = max(0.0, (datetime.now(timezone.utc) - updated).total_seconds())
        if age > eta:
            return 'ETA: szacowany czas minął; transkrypcja trwa'
        eta -= age
    except (KeyError, TypeError, ValueError):
        # Older or incomplete job timestamps leave the last estimate unchanged.
        return f'ETA: około {short_time(eta)}'
    return f'ETA: około {short_time(eta)}'


def header(title, subtitle):
    ui.label(title).classes('page-title')
    ui.label(subtitle).classes('muted q-mt-xs q-mb-lg')


def option_form(settings, include_paths=False):
    fields = {}
    with ui.row().classes('w-full gap-4 items-start'):
        fields['model'] = ui.select({name: name for name in MODEL_REPOS}, label='Model Whisper', value=settings.model).classes('grow min-w-44').props('outlined')
        fields['language'] = ui.select({'': 'Wykryj automatycznie', 'pl': 'Polski', 'en': 'Angielski', 'de': 'Niemiecki', 'es': 'Hiszpański', 'fr': 'Francuski', 'uk': 'Ukraiński'}, label='Język nagrania', value=settings.language).classes('grow min-w-44').props('outlined')
        fields['task'] = ui.select({'transcribe': 'Transkrypcja', 'translate': 'Tłumaczenie na angielski'}, label='Tryb', value=settings.task).classes('grow min-w-44').props('outlined')
    with ui.row().classes('w-full gap-4 items-start'):
        fields['device'] = ui.select({'cuda': 'GPU · CUDA', 'cpu': 'CPU (świadomie)'}, label='Urządzenie', value=settings.device).classes('grow min-w-44').props('outlined')
        fields['compute_type'] = ui.select({'auto': 'Automatyczna', 'float16': 'Wydajność · float16', 'int8_float16': 'Oszczędzanie VRAM · int8_float16'}, label='Precyzja GPU', value=settings.compute_type).classes('grow min-w-44').props('outlined')
        fields['beam_size'] = ui.number('Beam size', min=1, max=10, step=1, value=settings.beam_size, precision=0).classes('w-32').props('outlined')
        fields['temperature'] = ui.number('Temperature', min=0, max=1, step=.1, value=settings.temperature).classes('w-32').props('outlined')
    with ui.row().classes('w-full gap-5'):
        fields['vad'] = ui.switch('Filtr ciszy VAD', value=settings.vad)
        fields['word_timestamps'] = ui.switch('Znaczniki słów', value=settings.word_timestamps)
        fields['keep_model'] = ui.switch('Pozostaw model w pamięci', value=settings.keep_model)
    fields['initial_prompt'] = ui.textarea('Kontekst początkowy (opcjonalny)', value=settings.initial_prompt).classes('w-full').props('outlined autogrow')
    ui.label('Whisper tłumaczy mowę wyłącznie na angielski. Turbo nie obsługuje trybu tłumaczenia.').classes('hint')
    if include_paths:
        fields['cache_dir'] = ui.input('Katalog modeli', value=settings.cache_dir).classes('w-full').props('outlined')
        fields['output_dir'] = ui.input('Katalog eksportów', value=settings.output_dir).classes('w-full').props('outlined')
        with ui.row().classes('w-full gap-4'):
            fields['workers'] = ui.number('Równoległe zadania', min=1, max=2, step=1, precision=0, value=settings.workers).props('outlined')
            fields['max_upload_gb'] = ui.number('Limit pliku (GB)', min=1, max=100, step=1, precision=0, value=settings.max_upload_gb).props('outlined')
            fields['theme'] = ui.select({'dark': 'Ciemny', 'light': 'Jasny'}, label='Motyw', value=settings.theme).classes('w-40').props('outlined')
        ui.label('Na GPU wykonywane jest najwyżej jedno zadanie naraz.').classes('hint')
    return fields


def form_values(fields, original):
    return Settings.model_validate({**original.model_dump(), **{key: field.value for key, field in fields.items()}})


class Interface:
    def __init__(self, db, models, queue, recorder):
        self.db, self.model_manager, self.tasks, self.recorder = db, models, queue, recorder
        self.timers = []
        self.tracked = set()
        self.staged = []
        self.page = 'dashboard'
        self.dark = ui.dark_mode(value=db.settings().theme == 'dark')
        ui.add_head_html(CSS)
        with ui.row().classes('shell no-wrap'):
            with ui.column().classes('sidebar no-wrap'):
                with ui.row().classes('items-center gap-2 q-mb-xl'):
                    ui.icon('graphic_eq', size='25px').classes('brand-icon')
                    with ui.column().classes('gap-0 brand-sub'):
                        ui.label('WhysperGUI').classes('brand-title')
                        ui.label('LOCAL STUDIO').classes('eyebrow')
                ui.label('PRZESTRZEŃ ROBOCZA').classes('eyebrow side-caption q-mb-sm')
                self.nav_buttons = {}
                for key, label, icon in NAV:
                    self.nav_buttons[key] = ui.button(label, icon=icon, on_click=lambda _, k=key: self.navigate(k)).props('flat no-caps align=left').classes('nav-btn')
                ui.space()
                ui.label('●  W pełni lokalnie').classes('hint side-caption')
            self.content = ui.column().classes('main')
        self.navigate('dashboard')
        ui.timer(2.0, self.check_completed)

    def timer(self, seconds, callback):
        self.timers.append(ui.timer(seconds, callback))

    def navigate(self, key, job_id=None):
        for timer in self.timers:
            timer.cancel()
        self.timers.clear()
        self.page = key
        for name, button in self.nav_buttons.items():
            button.classes(add='active' if name == key else '', remove='' if name == key else 'active')
        self.content.clear()
        with self.content:
            if key in ('editor', 'live'):
                getattr(self, key)(job_id)
            else:
                getattr(self, key)()

    def check_completed(self):
        for job_id in list(self.tracked):
            job = self.db.get(job_id)
            if not job or job['status'] in ('error', 'cancelled'):
                self.tracked.discard(job_id)
                if job and job['status'] == 'error':
                    ui.notify(f"Błąd: {job['name']} — {job['message']}", type='negative', timeout=8000)
            elif job['status'] == 'done':
                self.tracked.discard(job_id)
                ui.notify('Transkrypcja jest gotowa.', type='positive')
                self.navigate('editor', job_id)
                break

    def dashboard(self):
        header('Twoje centrum transkrypcji', 'Wszystko, czego potrzebujesz do pracy z nagraniami — lokalnie i pod kontrolą.')
        with ui.row().classes('w-full gap-4'):
            def metric(label, icon):
                with ui.card().classes('panel metric'):
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label(label).classes('muted text-sm')
                        ui.icon(icon, size='21px').classes('muted')
                    value = ui.label('—').classes('metric-value q-mt-sm')
                    sub = ui.label('').classes('hint')
                    return value, sub
            gpu, gpu_sub = metric('Karta graficzna', 'memory')
            vram, vram_sub = metric('Pamięć VRAM', 'donut_large')
            waiting, waiting_sub = metric('W kolejce', 'hourglass_empty')
            completed, completed_sub = metric('Ukończone', 'task_alt')
        with ui.row().classes('w-full gap-4 q-mt-md'):
            with ui.card().classes('panel grow'):
                with ui.row().classes('w-full justify-between items-center'):
                    ui.label('Bieżąca aktywność').classes('section-title')
                    ui.button('Otwórz kolejkę', icon='arrow_forward', on_click=lambda: self.navigate('queue')).props('flat no-caps')
                activity = ui.label('').classes('muted q-mt-md')
                bar = ui.linear_progress(value=0, show_value=False).classes('w-full q-mt-sm')
                activity_eta = ui.label('').classes('hint')
                active_job = {'id': None}
                preview = ui.button('Podgląd na żywo', icon='subtitles',
                                    on_click=lambda: self.navigate('live', active_job['id'])).props('flat no-caps')
                preview.set_enabled(False)
            with ui.card().classes('panel grow'):
                ui.label('Zasoby i model').classes('section-title')
                resources = ui.label('').classes('muted q-mt-md whitespace-pre-line')
                ui.button('Diagnostyka', icon='monitor_heart', on_click=lambda: self.navigate('diagnostics')).props('flat no-caps')
        with ui.card().classes('panel w-full q-mt-md'):
            with ui.row().classes('w-full justify-between items-center'):
                with ui.column().classes('gap-1'):
                    ui.label('Gotowy na kolejne nagranie?').classes('section-title')
                    ui.label('Dodaj plik audio lub wideo, wybierz model i rozpocznij pracę.').classes('muted')
                ui.button('Nowa transkrypcja', icon='add', on_click=lambda: self.navigate('transcription')).props('unelevated no-caps')

        def refresh():
            stats = gpu_stats()
            gpu.set_text(stats.get('name', 'Niedostępna').replace('NVIDIA GeForce ', ''))
            gpu_sub.set_text(f"Obciążenie {stats.get('utilization', '—')}% · {stats.get('temperature', '—')}°C" if 'error' not in stats else stats['error'])
            vram.set_text(human_size(stats.get('memory_used')))
            vram_sub.set_text('z ' + human_size(stats.get('memory_total')))
            jobs = self.db.list()
            waiting.set_text(str(sum(j['status'] == 'waiting' for j in jobs)))
            waiting_sub.set_text('oczekujących nagrań')
            completed.set_text(str(sum(j['status'] == 'done' for j in jobs)))
            completed_sub.set_text('w historii')
            active = next((j for j in jobs if j['status'] in ('downloading', 'preparing', 'transcribing', 'exporting')), None)
            activity.set_text(f"{active['name']} · {STATUS[active['status']]}" if active else 'Brak aktywnych zadań')
            bar.value = active['progress'] or 0 if active else 0
            activity_eta.set_text(eta_text(active) if active else '')
            active_job['id'] = active['id'] if active else None
            preview.set_enabled(bool(active))
            resources.set_text(f"RAM: {human_size(psutil.virtual_memory().used)} / {human_size(psutil.virtual_memory().total)}\nZaładowany model GPU: {self.tasks.gpu_engine.loaded_name or 'brak'}")
        self.timer(2, refresh)

    def transcription(self):
        header('Nowa transkrypcja', 'Dodaj nagrania i dostosuj sposób rozpoznawania mowy.')
        settings = self.db.settings()
        with ui.card().classes('panel w-full'):
            ui.label('1  ·  Wybierz nagrania').classes('section-title')
            ui.label('MP3, WAV, FLAC, M4A, AAC, OGG, OPUS, MP4, MKV, MOV, AVI lub WEBM').classes('hint')
            files_area = ui.column().classes('w-full gap-1')

            def refresh_files():
                files_area.clear()
                with files_area:
                    for item in self.staged:
                        ui.label(f"{item['name']}  ·  {human_size(item['size'])}  ·  {short_time(item['duration'])}").classes('text-sm')
            async def upload(e):
                try:
                    target, size, duration = await run.io_bound(store_upload, e.content, e.name, settings.max_upload_gb * 1024**3)
                    self.staged.append({'name': Path(e.name).name, 'path': str(target), 'size': size, 'duration': duration})
                    refresh_files()
                except Exception as exc:
                    ui.notify(str(exc), type='negative', timeout=8000)
            ui.upload(label='Przeciągnij pliki tutaj lub wybierz z dysku', multiple=True, auto_upload=True,
                      max_file_size=settings.max_upload_gb * 1024**3, on_upload=upload,
                      on_rejected=lambda: ui.notify('Plik przekracza limit lub został odrzucony.', type='negative')).classes('w-full').props('accept=".mp3,.wav,.flac,.m4a,.aac,.ogg,.opus,.mp4,.mkv,.mov,.avi,.webm"')
            refresh_files()
        with ui.card().classes('panel w-full q-mt-md'):
            ui.label('2  ·  Ustawienia transkrypcji').classes('section-title q-mb-sm')
            fields = option_form(settings)
            def submit():
                try:
                    options = form_values(fields, settings)
                    if not self.staged:
                        raise ValueError('Dodaj co najmniej jedno nagranie.')
                    for item in self.staged:
                        job_id = self.db.add_job(item['name'], item['path'], options)
                        self.tracked.add(job_id)
                    count = len(self.staged)
                    self.staged.clear()
                    ui.notify(f'Dodano do kolejki: {count}', type='positive')
                    self.navigate('queue')
                except Exception as exc:
                    ui.notify(str(exc), type='negative')
            ui.button('Dodaj do kolejki', icon='arrow_forward', on_click=submit).props('unelevated no-caps').classes('q-mt-md')

    def recording(self):
        header('Nagrywanie', 'Nagraj dźwięk z mikrofonu podłączonego do tego komputera.')
        with ui.card().classes('panel w-full'):
            try:
                devices = self.recorder.devices()
            except Exception as exc:
                devices = {}
                ui.label(f'Nie można odczytać urządzeń: {exc}').classes('text-negative')
            device = ui.select(devices, label='Mikrofon', value=next(iter(devices), None)).classes('w-full').props('outlined')
            ui.label('●').classes('record-dot')
            clock = ui.label('00:00:00').classes('metric-value')
            level = ui.linear_progress(value=0, show_value=False).classes('w-full')
            state = ui.label('').classes('hint')
            playback = ui.column().classes('w-full')
            def start():
                try:
                    if device.value is None:
                        raise ValueError('Nie znaleziono mikrofonu wejściowego.')
                    self.recorder.start(int(device.value))
                    playback.clear()
                except Exception as exc:
                    ui.notify(str(exc), type='negative')
            def pause():
                try:
                    self.recorder.pause()
                except Exception as exc:
                    ui.notify(str(exc), type='negative')
            def stop():
                try:
                    path = self.recorder.stop()
                    playback.clear()
                    with playback:
                        ui.audio(f'/recordings/{path.name}').classes('w-full')
                        def send():
                            job_id = self.db.add_job(path.name, str(path), self.db.settings())
                            self.tracked.add(job_id)
                            self.navigate('queue')
                        with ui.row():
                            ui.button('Transkrybuj nagranie', icon='graphic_eq', on_click=send).props('unelevated no-caps')
                            ui.button('Pobierz WAV', icon='download', on_click=lambda: ui.download(f'/recordings/{path.name}')).props('flat no-caps')
                except Exception as exc:
                    ui.notify(str(exc), type='negative')
            with ui.row().classes('q-mt-md'):
                ui.button('Rozpocznij', icon='fiber_manual_record', on_click=start).props('unelevated no-caps')
                ui.button('Pauza / wznów', icon='pause', on_click=pause).props('outline no-caps')
                ui.button('Zakończ', icon='stop', on_click=stop).props('outline no-caps')
            def refresh():
                status = self.recorder.status()
                clock.set_text(short_time(status['seconds']))
                level.value = min(1, status['level'])
                state.set_text(('Pauza' if status['paused'] else 'Nagrywanie') if status['active'] else 'Gotowość')
                if status['error']:
                    state.set_text(state.text + ' · ' + status['error'])
            self.timer(.25, refresh)

    def queue(self):
        header('Kolejka zadań', 'Śledź przetwarzanie, zmieniaj kolejność i zarządzaj zadaniami.')
        area = ui.column().classes('w-full')

        def draw():
            area.clear()
            jobs = self.db.list(('waiting', 'downloading', 'preparing', 'transcribing', 'exporting', 'error', 'cancelled'))
            with area:
                with ui.card().classes('panel w-full'):
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label('Zadania').classes('section-title')
                        ui.label(f'{len(jobs)} pozycji').classes('hint')
                    if not jobs:
                        ui.label('Kolejka jest pusta. Dodaj nagranie, aby rozpocząć.').classes('muted q-my-md')
                    for job in jobs:
                        with ui.column().classes('w-full job-row gap-1'):
                            with ui.row().classes('w-full justify-between items-center no-wrap queue-job-header'):
                                with ui.column().classes('gap-0 grow'):
                                    ui.label(job['name']).classes('font-medium')
                                    info = STATUS[job['status']]
                                    if job['duration']:
                                        info += ' · ' + short_time(job['duration'])
                                    if job['elapsed']:
                                        info += f" · czas {short_time(job['elapsed'])}"
                                    ui.label(info).classes('hint')
                                ui.label(STATUS[job['status']]).classes('status-chip')
                                if job['status'] not in ('error', 'cancelled'):
                                    ui.button('Podgląd na żywo', icon='subtitles',
                                              on_click=lambda _, j=job['id']: self.navigate('live', j)).props('flat no-caps')
                                if job['status'] == 'waiting':
                                    ui.button(icon='arrow_upward', on_click=lambda _, j=job['id']: (self.db.move(j, -1), draw())).props('flat round dense').tooltip('Przesuń w górę')
                                    ui.button(icon='arrow_downward', on_click=lambda _, j=job['id']: (self.db.move(j, 1), draw())).props('flat round dense').tooltip('Przesuń w dół')
                                if job['status'] in ('error', 'cancelled'):
                                    ui.button('Ponów', icon='replay', on_click=lambda _, j=job['id']: self.retry(j, draw)).props('flat no-caps')
                                else:
                                    ui.button(icon='close', on_click=lambda _, j=job['id']: (self.db.cancel(j), draw())).props('flat round dense').tooltip('Anuluj')
                            if job['message']:
                                ui.label(job['message']).classes('text-negative text-sm break-all' if job['status'] == 'error' else 'hint')
                            if job['status'] == 'downloading':
                                options = Settings.model_validate_json(job['options'])
                                detail = self.model_manager.details(options.model, options.cache_dir)
                                if detail.get('total'):
                                    ui.linear_progress(value=(detail.get('downloaded') or 0) / detail['total'], show_value=False).classes('w-full')
                                    ui.label(f"Pobrano {human_size(detail.get('downloaded'))} / {human_size(detail['total'])}").classes('hint')
                            if job['progress'] is not None and job['status'] == 'transcribing':
                                ui.linear_progress(value=job['progress'], show_value=False).classes('w-full')
                                ui.label(f"{job['progress'] * 100:.1f}% nagrania · {eta_text(job)}").classes('hint')
                            elif job['status'] == 'transcribing':
                                ui.label(eta_text(job)).classes('hint')
        self.timer(2, draw)

    def live(self, job_id):
        job = self.db.get(job_id)
        if not job:
            header('Nie znaleziono zadania', 'Wybierz zadanie z kolejki.')
            return
        header('Podgląd na żywo', job['name'])
        with ui.card().classes('panel w-full'):
            with ui.row().classes('w-full items-center justify-between'):
                status = ui.label('').classes('section-title')
                ui.button('Wróć do kolejki', icon='arrow_back', on_click=lambda: self.navigate('queue')).props('flat no-caps')
            progress = ui.linear_progress(value=0, show_value=False).classes('w-full')
            progress_info = ui.label('').classes('hint')
            eta = ui.label('').classes('text-sm')
            ui.label('Tekst pojawia się po rozpoznaniu kolejnych segmentów; wynik końcowy można edytować po zakończeniu.').classes('hint')
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Rozpoznany tekst').classes('section-title')
                follow = ui.switch('Przewijaj do najnowszych', value=True)
            with ui.scroll_area().classes('live-feed w-full').style('height: 480px') as scroll:
                with ui.column().classes('w-full q-pa-md gap-3') as feed:
                    empty = ui.label('Oczekiwanie na pierwszy rozpoznany segment…').classes('muted')
            texts = []
            last_sequence = -1
            with ui.row().classes('w-full'):
                def copy():
                    ui.clipboard.write('\n'.join(texts))
                    ui.notify('Skopiowano dotychczasową transkrypcję.')
                copy_button = ui.button('Kopiuj tekst', icon='content_copy', on_click=copy).props('flat no-caps')
                copy_button.set_enabled(False)
                result_button = ui.button('Otwórz gotowy wynik', icon='open_in_new',
                                          on_click=lambda: self.navigate('editor', job_id)).props('unelevated no-caps')
                result_button.set_visibility(False)

            def refresh():
                nonlocal last_sequence
                current = self.db.get(job_id)
                if not current:
                    status.set_text('Nie znaleziono zadania.')
                    return
                status.set_text(STATUS.get(current['status'], current['status']))
                progress.value = current['progress'] or 0
                progress_info.set_text(f"Przetworzono {current['progress'] * 100:.1f}% nagrania" if current['progress'] is not None else 'Czas trwania lub postęp nie jest jeszcze dostępny.')
                eta.set_text(eta_text(current))
                result_button.set_visibility(current['status'] == 'done')
                rows = self.db.live_segments(job_id, after=last_sequence)
                if rows:
                    empty.set_visibility(False)
                    lines = []
                    for row in rows:
                        lines.append(f"[{short_time(row['start'])}] {row['text']}")
                        texts.append(row['text'])
                    last_sequence = rows[-1]['sequence']
                    with feed:
                        ui.label('\n'.join(lines)).classes('w-full whitespace-pre-wrap break-words')
                    copy_button.set_enabled(True)
                    if follow.value:
                        scroll.scroll_to(percent=1)
                elif not texts and current['status'] == 'done':
                    empty.set_text('Zadanie zakończone. Otwórz wynik, aby zobaczyć pełną transkrypcję.')
                elif not texts and current['status'] in ('error', 'cancelled'):
                    empty.set_text('Zadanie zakończono przed rozpoznaniem pierwszego segmentu.')
            self.timer(1, refresh)

    def retry(self, job_id, draw):
        self.db.retry(job_id)
        self.tracked.add(job_id)
        draw()

    def history(self):
        header('Historia', 'Wróć do ukończonych transkrypcji i pobierz swoje wyniki.')
        with ui.card().classes('panel w-full'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Twoje transkrypcje').classes('section-title')
                search = ui.input(placeholder='Szukaj po nazwie…').props('outlined dense clearable').classes('w-64')
            area = ui.column().classes('w-full')
            def draw():
                area.clear()
                query = (search.value or '').casefold()
                jobs = [j for j in self.db.list(('done',)) if query in j['name'].casefold()]
                with area:
                    if not jobs:
                        ui.label('Brak wyników.').classes('muted q-my-md')
                    for job in jobs:
                        with ui.row().classes('w-full job-row items-center no-wrap'):
                            with ui.column().classes('grow gap-0'):
                                ui.label(job['name']).classes('font-medium')
                                ui.label(f"{job['created_at'][:16].replace('T', ' ')} UTC · {short_time(job['duration'])} · {job['language'] or '—'}").classes('hint')
                            ui.button('Otwórz', icon='open_in_new', on_click=lambda _, j=job['id']: self.navigate('editor', j)).props('flat no-caps')
                            ui.button('TXT', on_click=lambda _, j=job['id']: ui.download(f'/export/{j}/txt')).props('flat no-caps')
                            ui.button(icon='delete_outline', on_click=lambda _, j=job['id']: confirm_delete(j)).props('flat round').tooltip('Usuń wpis z historii')
            def confirm_delete(job_id):
                with ui.dialog() as dialog, ui.card().classes('panel'):
                    ui.label('Usunąć ten wpis z historii?').classes('section-title')
                    ui.label('Zapisane wcześniej pliki eksportu pozostaną na dysku.').classes('hint')
                    with ui.row():
                        ui.button('Anuluj', on_click=dialog.close).props('flat no-caps')
                        def delete():
                            self.db.delete(job_id)
                            dialog.close()
                            draw()
                        ui.button('Usuń wpis', on_click=delete, color='negative').props('flat no-caps')
                dialog.open()
            search.on_value_change(lambda _: draw())
            draw()

    def models(self):
        header('Modele AI', 'Pobierz modele raz i korzystaj z nich również bez internetu.')
        settings = self.db.settings()
        ui.label(f'Katalog modeli: {settings.cache_dir}').classes('hint q-mb-md break-all')
        with ui.card().classes('panel w-full'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Dostępność modeli').classes('section-title')
                summary = ui.label('').classes('hint')
            elements = {}
            for name in MODEL_REPOS:
                with ui.row().classes('w-full model-row items-center no-wrap') as model_row:
                    with ui.column().classes('grow gap-0'):
                        ui.label(name).classes('font-medium text-base')
                        info = ui.label('').classes('hint')
                        bar = ui.linear_progress(value=0, show_value=False).classes('w-full')
                    with ui.row().classes('model-badge items-center no-wrap') as badge:
                        badge_icon = ui.icon('cloud_download', size='17px')
                        badge_text = ui.label('NIEPOBRANY')
                    download = ui.button('Pobierz', icon='download', on_click=lambda _, n=name: self.models_start(n, settings)).props('flat no-caps')
                    async def remove(n=name):
                        try:
                            active = self.db.list(('downloading', 'preparing', 'transcribing', 'exporting'))
                            if any(Settings.model_validate_json(j['options']).model == n for j in active):
                                raise RuntimeError('Model jest używany przez aktywne zadanie.')
                            await run.io_bound(self.tasks.gpu_engine.unload)
                            await run.io_bound(self.model_manager.delete, n, settings.cache_dir)
                            ui.notify('Usunięto model.', type='positive')
                        except Exception as exc:
                            ui.notify(str(exc), type='negative')
                    delete = ui.button(icon='delete_outline', on_click=remove).props('flat round').tooltip('Usuń model')
                    elements[name] = (model_row, info, bar, badge, badge_icon, badge_text, download, delete)
            def refresh():
                available_count = 0
                for name, (row, info, bar, badge, icon, badge_text, button, delete) in elements.items():
                    detail = self.model_manager.details(name, settings.cache_dir)
                    available = detail['available']
                    available_count += int(available)
                    running = detail.get('status') in ('Pobieranie', 'Sprawdzanie plików')
                    suffix = f" · {detail['status']}" if detail.get('status') and detail['status'] != 'Pobrano' else ''
                    if detail.get('total'):
                        suffix += f" {human_size(detail.get('downloaded'))} / {human_size(detail['total'])}"
                    info.set_text(f"Na dysku: {human_size(detail['size'])} · rozmiar modelu {detail['estimated']}" + suffix)
                    row.classes(add='ready' if available else '', remove='' if available else 'ready')
                    badge.classes(add='ready' if available else '', remove='' if available else 'ready')
                    icon.set_name('check_circle' if available else 'cloud_download')
                    badge_text.set_text('POBRANY · GOTOWY OFFLINE' if available else ('POBIERANIE' if running else 'NIEPOBRANY'))
                    bar.value = (detail.get('downloaded') or 0) / detail['total'] if detail.get('total') else 0
                    bar.set_visibility(bool(running and detail.get('total')))
                    button.set_text('Pobrano' if available else 'Pobierz')
                    button.set_enabled(not available and not running)
                    delete.set_enabled(bool(detail['size']) and not running)
                summary.set_text(f'Pobrane: {available_count} z {len(elements)} modeli')
            refresh()
            self.timer(1.5, refresh)

    def models_start(self, name, settings):
        self.model_manager.start_download(name, settings.cache_dir)
        ui.notify(f'Pobieranie modelu {name} rozpoczęte.')

    def editor(self, job_id):
        job = self.db.get(job_id)
        if not job or not job['result']:
            header('Nie znaleziono wyniku', 'Wybierz ukończoną transkrypcję z historii.')
            return
        result = json.loads(job['result'])
        header('Edytor transkrypcji', job['name'])
        with ui.row().classes('w-full gap-3 items-center q-mb-md'):
            ui.label(f"{result.get('model', '—')} · {result.get('device', '—').upper()} · {result.get('language', '—')} · RTF {result.get('rtf', 0) or 0:.2f}").classes('hint grow')
            for fmt in ('txt', 'srt', 'vtt', 'json', 'csv'):
                ui.button(fmt.upper(), on_click=lambda _, f=fmt: ui.download(f'/export/{job_id}/{f}')).props('outline no-caps dense')
            async def reexport():
                try:
                    current = self.db.get(job_id)
                    await run.io_bound(export_all, json.loads(current['result']), Settings.model_validate_json(current['options']).output_dir, job_id)
                    ui.notify('Zaktualizowano wszystkie pliki eksportu.', type='positive')
                except Exception as exc:
                    ui.notify(str(exc), type='negative')
            ui.button('Ponów eksport', icon='save_alt', on_click=reexport).props('flat no-caps')
        with ui.card().classes('panel w-full'):
            ext = Path(job['name']).suffix.lower()
            player = ui.video(f'/media/{job_id}').classes('w-full') if ext in ('.mp4', '.mkv', '.mov', '.avi', '.webm') else ui.audio(f'/media/{job_id}').classes('w-full')
            with ui.row().classes('w-full items-center'):
                search = ui.input(placeholder='Szukaj w transkrypcji…').props('outlined dense clearable').classes('grow')
                def copy():
                    current = json.loads(self.db.get(job_id)['result'])
                    ui.clipboard.write('\n'.join(s['text'] for s in current['segments']))
                    ui.notify('Skopiowano transkrypcję.')
                ui.button('Kopiuj całość', icon='content_copy', on_click=copy).props('flat no-caps')
            ui.label('Zmiany zapisują się automatycznie. Kliknij znacznik czasu, aby przejść do fragmentu.').classes('hint')
            rows = ui.column().classes('w-full gap-0')
            def draw():
                rows.clear()
                current = json.loads(self.db.get(job_id)['result'])
                query = (search.value or '').casefold()
                with rows:
                    for index, segment in enumerate(current['segments']):
                        if query not in segment['text'].casefold():
                            continue
                        with ui.row().classes('w-full job-row no-wrap items-start'):
                            def seek(second=segment['start']):
                                player.seek(float(second))
                                player.play()
                            ui.button(short_time(segment['start']), on_click=seek).props('flat no-caps dense').classes('text-primary')
                            def save(e, idx=index):
                                try:
                                    self.db.edit_segment(job_id, idx, e.value or '')
                                except Exception as exc:
                                    ui.notify(str(exc), type='negative')
                            ui.textarea(value=segment['text'], on_change=save).classes('grow').props('outlined autogrow debounce=700')
                            if segment.get('words'):
                                ui.label(' '.join(f"{w['word'].strip()} [{short_time(w['start'])}]" for w in segment['words'])).classes('hint')
            search.on_value_change(lambda _: draw())
            draw()

    def settings(self):
        header('Ustawienia', 'Wybierz ustawienia domyślne i lokalizacje plików.')
        settings = self.db.settings()
        with ui.card().classes('panel w-full'):
            fields = option_form(settings, include_paths=True)
            def save():
                try:
                    updated = form_values(fields, settings)
                    for path in (updated.cache_dir, updated.output_dir):
                        Path(path).expanduser().mkdir(parents=True, exist_ok=True)
                    self.db.save_settings(updated)
                    self.dark.value = updated.theme == 'dark'
                    ui.notify('Zapisano ustawienia.', type='positive')
                except Exception as exc:
                    ui.notify(str(exc), type='negative')
            ui.button('Zapisz ustawienia', icon='save', on_click=save).props('unelevated no-caps').classes('q-mt-md')

    def diagnostics(self):
        header('Diagnostyka', 'Sprawdź zgodność środowiska i wykonaj rzeczywistą inferencję na GPU.')
        with ui.card().classes('panel w-full'):
            ui.label('Środowisko').classes('section-title')
            summary = ui.label('Odczytywanie informacji…').classes('whitespace-pre-wrap text-sm w-full break-all')
            async def refresh():
                data = await run.io_bound(diagnostics)
                summary.set_text(json.dumps(data, ensure_ascii=False, indent=2))
            async def test_gpu():
                summary.set_text('Trwa rzeczywisty test Faster-Whisper na CUDA…')
                settings = self.db.settings()
                path = self.model_manager.path('tiny', settings.cache_dir)
                if not self.model_manager.available('tiny', settings.cache_dir):
                    summary.set_text('Brak modelu tiny. Pobierz go w zakładce Modele AI, a następnie ponów test.')
                    return
                result = await run.io_bound(inference_test, path)
                summary.set_text(json.dumps(result, ensure_ascii=False, indent=2))
                ui.notify('Test GPU zakończony.' if result.get('ok') else 'Test GPU nie powiódł się. Sprawdź szczegóły.', type='positive' if result.get('ok') else 'negative')
            with ui.row():
                ui.button('Odśwież', icon='refresh', on_click=refresh).props('outline no-caps')
                ui.button('Uruchom test CUDA', icon='play_arrow', on_click=test_gpu).props('unelevated no-caps')
            self.timers.append(ui.timer(.1, refresh, once=True))
        from app.config import LOGS
        with ui.expansion('Ostatnie wpisy dziennika', icon='article').classes('w-full q-mt-md'):
            log = LOGS / 'app.log'
            try:
                with log.open('rb') as stream:
                    stream.seek(0, 2)
                    stream.seek(max(0, stream.tell() - 12000))
                    tail = stream.read().decode('utf-8', errors='replace')
            except OSError:
                tail = 'Brak wpisów.'
            ui.code(tail, language='text').classes('w-full')
