from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import psutil
from nicegui import run, ui

from app.config import MODEL_REPOS, Settings
from app.core.audio_processor import VIDEO_EXTENSIONS, store_upload
from app.core.exporter import export_all, segment_text, timestamp
from app.core.gpu_manager import cuda_devices, diagnostics, gpu_stats, inference_test

STATUS = {'waiting': 'Oczekuje', 'downloading': 'Pobieranie modelu', 'preparing': 'Przygotowanie',
          'transcribing': 'Transkrypcja', 'exporting': 'Eksport', 'done': 'Ukończono',
          'error': 'Błąd', 'cancelled': 'Anulowano'}
NAV = [('dashboard', 'Dashboard', 'space_dashboard'), ('transcription', 'Transkrypcja', 'graphic_eq'),
       ('captions', 'Napisy na żywo', 'closed_caption'), ('recording', 'Nagrywanie', 'mic_none'), ('queue', 'Kolejka', 'view_list'),
       ('history', 'Historia', 'history'), ('models', 'Modele AI', 'memory'),
       ('settings', 'Ustawienia', 'tune'), ('diagnostics', 'Diagnostyka', 'monitor_heart')]
EDITOR_PAGE_SIZE = 50
UNKNOWN_SPEAKER = 'unknown:'


def speaker_filter_options(segments, selected=()):
    """Use distinct keys for unnamed speakers and user-entered names."""
    names = {segment.get('speaker') for segment in segments if segment.get('speaker')}
    names.update(key[len('speaker:'):] for key in selected if key.startswith('speaker:'))
    return {UNKNOWN_SPEAKER: 'Nieustalony',
            **{f'speaker:{name}': name for name in sorted(names, key=str.casefold)}}


def matching_segments(segments, query='', speakers=()):
    query = query.casefold()
    selected = set(speakers or ())
    return [(index, segment) for index, segment in enumerate(segments)
            if (not selected or (f"speaker:{segment['speaker']}" if segment.get('speaker') else UNKNOWN_SPEAKER) in selected)
            and (query in segment['text'].casefold() or query in (segment.get('speaker') or '').casefold())]

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
.model-row.missing { border-color:rgba(148,163,184,.35); }
.model-row.partial { border-color:rgba(251,191,36,.55); background:rgba(245,158,11,.06); }
.model-row.downloading { border-color:rgba(56,189,248,.7); background:rgba(14,165,233,.09); }
.model-row.ready { border-color:rgba(52,211,153,.6); background:rgba(16,185,129,.08); }
.model-row.error { border-color:rgba(248,113,113,.7); background:rgba(239,68,68,.08); }
.model-info { color:#b3bed0; font-size:12px; line-height:1.5; padding:3px 0; overflow-wrap:anywhere; }
.model-badge { padding:7px 11px; border-radius:8px; font-size:12px; font-weight:700; letter-spacing:.03em; }
.model-badge.missing { background:rgba(148,163,184,.16); color:#cbd5e1; }
.model-badge.partial { background:rgba(245,158,11,.18); color:#fcd34d; }
.model-badge.downloading { background:rgba(14,165,233,.22); color:#7dd3fc; }
.model-badge.ready { background:rgba(16,185,129,.2); color:#6ee7b7; }
.model-badge.error { background:rgba(239,68,68,.2); color:#fca5a5; }
.body--light .model-info { color:#475569; }
.body--light .model-badge.missing { color:#334155; }
.body--light .model-badge.partial { color:#92400e; }
.body--light .model-badge.downloading { color:#075985; }
.body--light .model-badge.ready { color:#047857; }
.body--light .model-badge.error { color:#b91c1c; }
@keyframes model-spin { to { transform:rotate(360deg); } }
.model-spin { animation:model-spin 1.2s linear infinite; }
@media (prefers-reduced-motion:reduce) { .model-spin { animation:none; } }
.live-feed { background:rgba(2,6,23,.24); border:1px solid var(--line); border-radius:10px; }
.body--light .live-feed { background:#f8fafc; }
.live-caption-text { font-size:22px; line-height:1.55; padding:20px; white-space:pre-wrap; overflow-wrap:anywhere; }
.live-microphone-active { color:#f87171; font-weight:700; }
.body--light .live-microphone-active { color:#b91c1c; }
.q-field--outlined .q-field__control { border-radius:9px; }
.q-btn { letter-spacing:0; }
.job-row { border-top:1px solid var(--line); padding:16px 0; }
.options-panel { border:1px solid var(--line); border-radius:10px; margin-top:10px; }
.parameter-guide-item { border-top:1px solid var(--line); padding:10px 0; }
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


def model_state(detail):
    status = detail.get('status') or ''
    if status == 'Sprawdzanie plików':
        return 'checking'
    if status == 'Pobieranie':
        return 'downloading'
    if status == 'Anulowanie':
        return 'cancelling'
    if status == 'Anulowano':
        return 'cancelled'
    if status.startswith('Błąd:'):
        return 'error'
    if detail.get('available'):
        return 'ready'
    return 'partial' if detail.get('size', 0) > 0 else 'missing'


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


def dashboard_gpu_id(jobs, default_settings, loaded_gpu):
    for status in ('transcribing', 'preparing', 'downloading', 'exporting'):
        for job in jobs:
            if job['status'] != status:
                continue
            try:
                options = Settings.model_validate_json(job['options'])
            except ValueError:
                continue
            if options.device == 'cuda':
                return options.gpu_id
    return loaded_gpu['id'] if loaded_gpu else default_settings.gpu_id


def header(title, subtitle):
    ui.label(title).classes('page-title')
    ui.label(subtitle).classes('muted q-mt-xs q-mb-lg')


def gpu_picker(selected=''):
    """Load CUDA devices without blocking the page while the driver initializes."""
    options = {'': 'Automatycznie · pierwsza karta CUDA'}
    if selected:
        options[selected] = 'Wyszukiwanie zapisanej karty…'
    state = {'enabled': True, 'available': False, 'detected': False, 'valid_ids': set(), 'count': 0}
    with ui.column().classes('grow min-w-44 gap-0'):
        field = ui.select(options, label='Karta GPU', value=selected).classes('w-full').props('outlined')
        field.set_enabled(False)
        hint = ui.label('Wykrywanie kart CUDA…').classes('hint')

    def set_enabled(enabled):
        state['enabled'] = enabled
        field.set_enabled(enabled and state['available'])

    def update_hint():
        if not state['detected']:
            return
        if field.value and field.value not in state['valid_ids']:
            hint.set_text('Zapisana karta nie jest dostępna. Wybierz inną kartę przed transkrypcją.')
        elif state['available']:
            hint.set_text(f"Wykryto kart CUDA: {state['count']}. Wybór jest zapisywany dla zadania.")
        else:
            hint.set_text('Nie wykryto kart CUDA. Wybierz CPU lub sprawdź sterownik NVIDIA.')

    field.on_value_change(lambda _: update_hint())

    async def detect():
        try:
            devices = await run.io_bound(cuda_devices)
            if field.is_deleted:
                return
            devices = devices or []
            current = field.value or ''
            choices = {'': 'Automatycznie · pierwsza karta CUDA'}
            for device in devices:
                memory = f" · {human_size(device['memory_total'])} VRAM" if device.get('memory_total') else ''
                choices[device['id']] = f"CUDA {device['index']} · {device['name']}{memory}"
            valid_ids = set(choices)
            if current.startswith('cuda:') and current[5:].isdigit() and current not in choices:
                indexed = next((device for device in devices if device['index'] == int(current[5:])), None)
                if indexed:
                    choices[current] = f"CUDA {indexed['index']} · {indexed['name']} (zapis według indeksu)"
                    valid_ids.add(current)
            missing = bool(current and current not in choices)
            if missing:
                choices[current] = f'Niedostępna karta: {current}'
            field.set_options(choices)
            state['available'] = bool(devices)
            state['detected'] = True
            state['valid_ids'] = valid_ids
            state['count'] = len(devices)
            set_enabled(state['enabled'])
            update_hint()
        except Exception as exc:
            if not field.is_deleted:
                hint.set_text(f'Nie można wykryć kart CUDA: {exc}')
                set_enabled(False)

    ui.timer(.1, detect, once=True)
    return field, set_enabled


def option_form(settings, include_paths=False):
    fields = {}
    with ui.row().classes('w-full gap-4 items-start'):
        fields['model'] = ui.select({name: name for name in MODEL_REPOS}, label='Model Whisper', value=settings.model).classes('grow min-w-44').props('outlined')
        fields['language'] = ui.select({'': 'Wykryj automatycznie', 'pl': 'Polski', 'en': 'Angielski', 'de': 'Niemiecki', 'es': 'Hiszpański', 'fr': 'Francuski', 'uk': 'Ukraiński'}, label='Język nagrania', value=settings.language).classes('grow min-w-44').props('outlined')
        fields['task'] = ui.select({'transcribe': 'Transkrypcja', 'translate': 'Tłumaczenie na angielski'}, label='Tryb', value=settings.task).classes('grow min-w-44').props('outlined')
    with ui.row().classes('w-full gap-4 items-start'):
        fields['device'] = ui.select({'cuda': 'GPU · CUDA', 'vulkan': 'GPU · Vulkan', 'cpu': 'CPU'}, label='Urządzenie', value=settings.device).classes('grow min-w-44').props('outlined')
        fields['gpu_id'], set_gpu_enabled = gpu_picker(settings.gpu_id)
        fields['vulkan_device'] = ui.number('Numer GPU Vulkan', min=0, max=31, step=1, precision=0, value=settings.vulkan_device).classes('w-40').props('outlined')
        fields['compute_type'] = ui.select({'auto': 'Automatyczna', 'float16': 'Wydajność · float16', 'int8_float16': 'Oszczędzanie VRAM · int8_float16'}, label='Precyzja GPU', value=settings.compute_type).classes('grow min-w-44').props('outlined')
        fields['beam_size'] = ui.number('Beam size', min=1, max=10, step=1, value=settings.beam_size, precision=0).classes('w-32').props('outlined')
        fields['temperature'] = ui.number('Temperature', min=0, max=1, step=.1, value=settings.temperature).classes('w-32').props('outlined')
    ui.label('Temperatura 0 używa Beam size; powyżej 0 działa losowanie i parametr Best of.').classes('hint')
    with ui.row().classes('w-full gap-5'):
        fields['vad'] = ui.switch('Filtr ciszy VAD', value=settings.vad)
        fields['diarization'] = ui.switch('Rozróżniaj rozmówców', value=settings.diarization)
        fields['word_timestamps'] = ui.switch('Znaczniki słów', value=settings.word_timestamps)
        fields['keep_model'] = ui.switch('Pozostaw model w pamięci', value=settings.keep_model)
    ui.label('Rozmówcy: analiza lokalna na CPU. Przy pierwszym użyciu zostaną pobrane modele (około 46 MB); wyniki można poprawić w edytorze.').classes('hint')
    ui.label('Vulkan używa whisper.cpp i oddzielnych modeli GGML. Część opcji faster-whisper i pozostawianie modelu w pamięci nie jest dostępna.').classes('hint')
    fields['initial_prompt'] = ui.textarea('Kontekst początkowy (opcjonalny)', value=settings.initial_prompt).classes('w-full').props('outlined autogrow')
    ui.label('Whisper tłumaczy mowę wyłącznie na angielski. Turbo nie obsługuje trybu tłumaczenia.').classes('hint')

    with ui.expansion('Dodatkowe parametry transkrypcji', caption='Sposób rozpoznawania, podpowiedzi i filtr ciszy', icon='tune').classes('w-full options-panel'):
        ui.label('Wartości domyślne są odpowiednie na początek. Nieaktywne pola nie wpływają na wybrany tryb.').classes('hint q-mb-sm')
        with ui.row().classes('w-full gap-4 items-start'):
            fields['best_of'] = ui.number('Best of', min=1, max=10, step=1, precision=0, value=settings.best_of).classes('grow min-w-44').props('outlined')
            fields['patience'] = ui.number('Cierpliwość (patience)', min=.1, max=5, step=.1, precision=1, value=settings.patience).classes('grow min-w-44').props('outlined')
            fields['repetition_penalty'] = ui.number('Kara powtórzeń', min=1, max=3, step=.1, precision=1, value=settings.repetition_penalty).classes('grow min-w-44').props('outlined')
            fields['no_repeat_ngram_size'] = ui.number('Blokada powtórzeń (tokeny)', min=0, max=10, step=1, precision=0, value=settings.no_repeat_ngram_size).classes('grow min-w-44').props('outlined')
        with ui.row().classes('w-full gap-4 items-start'):
            fields['no_speech_threshold'] = ui.number('Próg ciszy Whisper', min=0, max=1, step=.05, precision=2, value=settings.no_speech_threshold).classes('grow min-w-44').props('outlined')
            fields['vad_threshold'] = ui.number('Próg mowy VAD', min=.05, max=.95, step=.05, precision=2, value=settings.vad_threshold).classes('grow min-w-44').props('outlined')
            fields['vad_min_silence_ms'] = ui.number('Minimalna cisza VAD (ms)', min=100, max=10000, step=100, precision=0, value=settings.vad_min_silence_ms).classes('grow min-w-44').props('outlined')
            fields['vad_speech_pad_ms'] = ui.number('Zapas mowy VAD (ms)', min=0, max=1500, step=50, precision=0, value=settings.vad_speech_pad_ms).classes('grow min-w-44').props('outlined')
        fields['condition_on_previous_text'] = ui.switch('Używaj poprzedniego tekstu jako kontekstu', value=settings.condition_on_previous_text)
        fields['hotwords'] = ui.textarea('Słowa podpowiedzi (opcjonalnie)', value=settings.hotwords,
                                         placeholder='Np. nazwiska, nazwy firm lub trudne terminy').classes('w-full').props('outlined autogrow')

    guide = [
        ('Model Whisper', 'Większe modele wymagają więcej pamięci i czasu; mogą lepiej rozpoznawać trudną mowę. Turbo nie służy do tłumaczenia.'),
        ('Język i tryb', 'Wybierz język nagrania albo wykrywanie automatyczne z pierwszej porcji. Tryb tłumaczenia Whisper daje tekst po angielsku.'),
        ('Urządzenie, karta GPU i precyzja', 'Wybierz CUDA, Vulkan lub CPU. W Vulkan numer karty odpowiada numeracji whisper.cpp; format modelu określa precyzję. Opcje precyzji dotyczą CUDA.'),
        ('Beam size', 'Przy temperaturze 0 określa liczbę rozważanych wariantów. 1 jest zwykle szybsze, 5 to domyślna wartość; większa liczba może pomóc kosztem czasu, bez gwarancji poprawy.'),
        ('Temperature', '0 daje deterministyczny wybór z użyciem Beam size. Powyżej 0 włącza losowanie i Best of; wyniki mogą się różnić między uruchomieniami. Aplikacja używa jednej ustawionej temperatury.'),
        ('Best of', 'Liczba losowanych kandydatów ocenianych przy temperaturze powyżej 0. Więcej kandydatów może wydłużyć pracę; domyślnie 5.'),
        ('Cierpliwość (patience)', 'Przy temperaturze 0 określa, jak długo wyszukiwanie rozważa warianty. Wyższa wartość może zwiększyć czas pracy; domyślnie 1.'),
        ('Kara powtórzeń', 'Wartość powyżej 1 zniechęca do ponownego używania tych samych tokenów; zbyt wysoka może usuwać potrzebne powtórzenia. 1 wyłącza karę.'),
        ('Blokada powtórzeń', 'Zapobiega powtórzeniu sekwencji o podanej liczbie tokenów (części słów, niekoniecznie całych słów). 0 wyłącza blokadę; niska wartość może nadmiernie ograniczyć tekst.'),
        ('Próg ciszy Whisper', 'Wraz z oceną pewności pomaga uznać fragment za ciszę. Wyższy próg może zachować więcej cichej mowy, ale też więcej ciszy; domyślnie 0,6.'),
        ('Filtr ciszy VAD i próg mowy', 'VAD próbuje oddzielić mowę od ciszy przed transkrypcją. Niższy próg może wykryć cichszą mowę, ale też więcej szumu; domyślnie 0,5.'),
        ('Minimalna cisza VAD', 'Czas ciszy w milisekundach potrzebny do oddzielenia fragmentów mowy. Krótszy dzieli przy krótszych pauzach; domyślnie 2000 ms.'),
        ('Zapas mowy VAD', 'Dodatkowy czas przed i po wykrytej mowie. Zbyt mały może obciąć początek lub koniec wypowiedzi; domyślnie 400 ms.'),
        ('Poprzedni tekst jako kontekst', 'Przekazuje poprzedni wynik w ramach tej samej porcji audio. Wyłączenie może ograniczyć zapętlanie powtórzeń, ale pogorszyć spójność; aplikacja przetwarza porcje do 30 sekund.'),
        ('Kontekst początkowy i słowa podpowiedzi', 'Pomagają zasugerować temat, pisownię nazw i trudnych słów. Są wskazówką, nie gwarancją rozpoznania; kontekst jest podawany dla każdej porcji.'),
        ('Znaczniki słów', 'Dodają orientacyjny czas dla każdego słowa, co może wydłużyć transkrypcję.'),
        ('Rozróżnianie rozmówców', 'Oznacza głosy jako Rozmówca 1, 2 itd. Numery obowiązują w bieżącym nagraniu, a nie identyfikują osób po imieniu. Nakładające się lub krótkie wypowiedzi mogą pozostać nieustalone.'),
        ('Pozostaw model w pamięci', 'Przyspiesza kolejne zadania tym samym modelem, ale nadal zajmuje RAM lub VRAM.'),
    ]
    if include_paths:
        guide.append(('Równoległe zadania i limit pliku', 'Limit pracowników dotyczy kolejki, lecz jednocześnie działa najwyżej jedna transkrypcja GPU. Limit pliku określa maksymalny rozmiar pojedynczego importu.'))
    with ui.expansion('Legenda parametrów', caption='Objaśnienia Beam size, temperatury, VAD i pozostałych opcji', icon='help_outline').classes('w-full options-panel'):
        for title, description in guide:
            with ui.column().classes('w-full gap-1 parameter-guide-item'):
                ui.label(title).classes('font-medium text-sm')
                ui.label(description).classes('hint')

    def update_parameter_controls():
        set_gpu_enabled(fields['device'].value == 'cuda')
        vulkan = fields['device'].value == 'vulkan'
        fields['vulkan_device'].set_enabled(vulkan)
        fields['compute_type'].set_enabled(fields['device'].value == 'cuda')
        sampling = (fields['temperature'].value or 0) > 0
        fields['beam_size'].set_enabled(not sampling)
        fields['patience'].set_enabled(not sampling and not vulkan)
        fields['best_of'].set_enabled(sampling)
        for name in ('repetition_penalty', 'no_repeat_ngram_size', 'condition_on_previous_text', 'hotwords', 'keep_model'):
            fields[name].set_enabled(not vulkan)
        for name in ('vad_threshold', 'vad_min_silence_ms', 'vad_speech_pad_ms'):
            fields[name].set_enabled(bool(fields['vad'].value))

    fields['temperature'].on_value_change(lambda _: update_parameter_controls())
    fields['vad'].on_value_change(lambda _: update_parameter_controls())
    fields['device'].on_value_change(lambda _: update_parameter_controls())
    update_parameter_controls()

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
    values = {**original.model_dump(), **{key: field.value for key, field in fields.items()}}
    if values['device'] == 'cpu':
        values['compute_type'] = 'auto'
    if values['device'] == 'vulkan':
        values.update(compute_type='auto', patience=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0,
                      condition_on_previous_text=True, hotwords='', keep_model=False)
    return Settings.model_validate(values)


class Interface:
    def __init__(self, db, models, queue, recorder, captions):
        self.db, self.model_manager, self.tasks, self.recorder = db, models, queue, recorder
        self.captions_service = captions
        self.owner_id = ui.context.client.id
        ui.context.client.on_disconnect(lambda: self.captions_service.stop(self.owner_id, finish=False))
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
        if self.page == 'captions' and key != 'captions':
            self.captions_service.stop(self.owner_id, finish=False)
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
                if self.page != 'captions':
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
            gpu, gpu_sub = metric('Karta CUDA', 'memory')
            vram, vram_sub = metric('Pamięć VRAM CUDA', 'donut_large')
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

        async def refresh():
            jobs = self.db.list()
            active = next((j for j in jobs if j['status'] in ('downloading', 'preparing', 'transcribing', 'exporting')), None)
            selected_id = dashboard_gpu_id(jobs, self.db.settings(), self.tasks.gpu_engine.loaded_gpu)
            stats = await run.io_bound(gpu_stats, selected_id)
            if gpu.is_deleted:
                return
            if 'error' in stats:
                gpu.set_text('Niedostępna')
                gpu_sub.set_text(stats['error'])
            else:
                name = stats.get('name', 'GPU').replace('NVIDIA GeForce ', '')
                gpu.set_text(f"CUDA {stats['index']} · {name}")
                utilization = stats.get('utilization')
                temperature = stats.get('temperature')
                gpu_sub.set_text(f"Obciążenie {str(utilization) + '%' if utilization is not None else '—'} · {str(temperature) + '°C' if temperature is not None else '—'}")
            vram.set_text(human_size(stats.get('memory_used')))
            vram_sub.set_text('z ' + human_size(stats.get('memory_total')))
            waiting.set_text(str(sum(j['status'] == 'waiting' for j in jobs)))
            waiting_sub.set_text('oczekujących nagrań')
            completed.set_text(str(sum(j['status'] == 'done' for j in jobs)))
            completed_sub.set_text('w historii')
            activity.set_text(f"{active['name']} · {STATUS[active['status']]}" if active else 'Brak aktywnych zadań')
            bar.value = active['progress'] or 0 if active else 0
            activity_eta.set_text(eta_text(active) if active else '')
            active_job['id'] = active['id'] if active else None
            preview.set_enabled(bool(active))
            loaded = self.tasks.gpu_engine.loaded_gpu
            gpu_suffix = f" · CUDA {loaded['index']}" if loaded else ''
            resources.set_text(f"RAM: {human_size(psutil.virtual_memory().used)} / {human_size(psutil.virtual_memory().total)}\nZaładowany model GPU: {self.tasks.gpu_engine.loaded_name or 'brak'}{gpu_suffix}")
        self.timer(2, refresh)

    def transcription(self):
        header('Nowa transkrypcja', 'Dodaj nagrania i dostosuj sposób rozpoznawania mowy.')
        settings = self.db.settings()
        with ui.card().classes('panel w-full'):
            ui.label('Czas pracy zadań').classes('section-title')
            running_time = ui.label('Brak aktywnych zadań · Czas od uruchomienia: —').classes('hint whitespace-pre-line')
            ui.label('Licznik rusza po rozpoczęciu zadania i obejmuje przygotowanie oraz pobieranie modelu.').classes('hint')

        def refresh_running_time():
            jobs = self.db.list(('downloading', 'preparing', 'transcribing', 'exporting'))
            lines = [f"{job['name']} · {STATUS[job['status']]} · Czas od uruchomienia: {short_time(self.tasks.elapsed(job))}"
                     for job in jobs]
            running_time.set_text('\n'.join(lines) if lines else 'Brak aktywnych zadań · Czas od uruchomienia: —')

        refresh_running_time()
        self.timer(1, refresh_running_time)
        with ui.card().classes('panel w-full q-mt-md'):
            ui.label('1  ·  Wybierz nagrania').classes('section-title')
            ui.label('MP3, WAV, FLAC, M4A, AAC, OGG, OPUS, MP4, MKV, MOV, AVI lub WEBM').classes('hint')
            ui.label('Z wideo zostanie zapisane audio; lokalna kopia wideo zostanie usunięta. Oryginał pozostaje na dysku.').classes('hint')
            files_area = ui.column().classes('w-full gap-1')

            def refresh_files():
                files_area.clear()
                with files_area:
                    for item in self.staged:
                        ui.label(f"{item['name']}  ·  {human_size(item['size'])}  ·  {short_time(item['duration'])}").classes('text-sm')
            async def upload(e):
                try:
                    if Path(e.name).suffix.lower() in VIDEO_EXTENSIONS:
                        ui.notify('Przygotowywanie audio z wideo…')
                    target, size, duration, warning = await run.io_bound(store_upload, e.content, e.name, settings.max_upload_gb * 1024**3)
                    self.staged.append({'name': Path(e.name).name, 'path': str(target), 'size': size, 'duration': duration})
                    refresh_files()
                    if warning:
                        ui.notify(warning, type='warning', timeout=10000)
                except Exception as exc:
                    ui.notify(str(exc), type='negative', timeout=8000)
                finally:
                    e.content.close()
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

    def captions(self):
        header('Napisy na żywo', 'Lokalna transkrypcja dźwięku z mikrofonu w języku wypowiedzi.')
        settings = self.db.settings()
        try:
            devices = self.captions_service.devices()
        except Exception as exc:
            devices = {}
            ui.label(f'Nie można odczytać mikrofonów: {exc}').classes('text-negative')
        default_device = self.captions_service.default_device()
        if default_device not in devices:
            default_device = next(iter(devices), None)
        with ui.card().classes('panel w-full'):
            ui.label('Źródło i ustawienia').classes('section-title')
            ui.label('Dźwięk jest przetwarzany lokalnie. W Vulkan porcje są zapisywane w plikach tymczasowych i usuwane po przetworzeniu. Opuszczenie zakładki lub zamknięcie karty zatrzymuje mikrofon.').classes('hint')
            with ui.row().classes('w-full gap-4 items-start'):
                microphone = ui.select(devices, label='Mikrofon', value=default_device).classes('grow min-w-44').props('outlined')
                model = ui.select({name: name for name in MODEL_REPOS}, label='Model Whisper', value=settings.model).classes('grow min-w-44').props('outlined')
                language = ui.select({'': 'Wykryj automatycznie', 'pl': 'Polski', 'en': 'Angielski', 'de': 'Niemiecki',
                                      'es': 'Hiszpański', 'fr': 'Francuski', 'uk': 'Ukraiński'},
                                     label='Język mowy', value=settings.language).classes('grow min-w-44').props('outlined')
                interval = ui.select({3: '3 sekundy', 5: '5 sekund', 8: '8 sekund', 10: '10 sekund'},
                                     label='Długość porcji', value=5).classes('min-w-44').props('outlined')
            with ui.row().classes('w-full gap-4 items-start'):
                device = ui.select({'cuda': 'GPU · CUDA', 'vulkan': 'GPU · Vulkan', 'cpu': 'CPU'}, label='Urządzenie', value=settings.device).classes('grow min-w-44').props('outlined')
                gpu, set_gpu_enabled = gpu_picker(settings.gpu_id)
                vulkan_device = ui.number('Numer GPU Vulkan', min=0, max=31, step=1, precision=0, value=settings.vulkan_device).classes('w-40').props('outlined')
                precision = ui.select({'auto': 'Automatyczna', 'float16': 'Float16', 'int8_float16': 'Int8/float16'},
                                      label='Precyzja GPU', value=settings.compute_type).classes('grow min-w-44').props('outlined')
                vad = ui.switch('Filtr ciszy VAD', value=settings.vad)
                diarization = ui.switch('Rozróżniaj rozmówców', value=settings.diarization)
            ui.label('Wybierz wcześniej pobrany model. Napisy pojawiają się po każdej porcji; szybkość zależy od modelu i komputera. Mniejsze modele zwykle zmniejszają opóźnienie.').classes('hint')
            ui.label('Automatyczne wykrywanie języka działa osobno dla każdej porcji; przy krótkich wypowiedziach wybór języka ręcznie może być dokładniejszy.').classes('hint')
            ui.label('W trybie GPU inne zadania GPU czekają na zakończenie sesji napisów. Nowa sesja zastępuje poprzedni tekst w pamięci.').classes('hint')
            ui.label('Rozróżnianie rozmówców działa lokalnie na CPU i przy pierwszym użyciu pobiera około 46 MB modeli. Krótkie lub nakładające się głosy mogą być oznaczone błędnie.').classes('hint')
            with ui.row().classes('w-full items-center gap-3 q-mt-sm'):
                status = ui.label('Gotowość').classes('font-medium')
                level = ui.linear_progress(value=0, show_value=False, color='red-4').classes('w-40')
                start = ui.button('Uruchom napisy', icon='mic', color='primary').props('unelevated no-caps')
                stop = ui.button('Zatrzymaj', icon='stop', color='negative').props('outline no-caps')
            warning = ui.label('').classes('text-warning text-sm')

        with ui.card().classes('panel w-full q-mt-md'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Napisy').classes('section-title')
                with ui.row():
                    copy = ui.button('Kopiuj całość', icon='content_copy').props('flat no-caps')
                    download = ui.button('Pobierz TXT', icon='download').props('flat no-caps')
            ui.label('Widok pokazuje ostatnie 50 fragmentów; kopiowanie i pobieranie obejmują całą bieżącą sesję.').classes('hint')
            with ui.scroll_area().classes('live-feed w-full').style('height: 320px') as scroll:
                output = ui.label('Tutaj pojawią się napisy.').classes('live-caption-text w-full')

        last_revision = {'value': None}
        controls = (microphone, model, language, interval, device, vad, diarization)

        def update_controls(active):
            for control in controls:
                control.set_enabled(not active)
            precision.set_enabled(not active and device.value == 'cuda')
            set_gpu_enabled(not active and device.value == 'cuda')
            vulkan_device.set_enabled(not active and device.value == 'vulkan')

        def start_live():
            try:
                if microphone.value is None:
                    raise ValueError('Nie znaleziono mikrofonu wejściowego.')
                values = {**settings.model_dump(), 'task': 'transcribe',
                    'model': model.value, 'language': language.value, 'device': device.value,
                    'gpu_id': gpu.value or '', 'compute_type': precision.value if device.value == 'cuda' else 'auto',
                    'vulkan_device': vulkan_device.value, 'vad': vad.value, 'diarization': diarization.value}
                if device.value == 'vulkan':
                    values.update(patience=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0,
                                  condition_on_previous_text=True, hotwords='', keep_model=False)
                options = Settings.model_validate(values)
                self.captions_service.start(int(microphone.value), options, int(interval.value), self.owner_id)
                refresh()
            except Exception as exc:
                ui.notify(str(exc), type='negative', timeout=8000)

        def stop_live():
            self.captions_service.stop(self.owner_id, finish=True)
            refresh()

        def copy_text():
            ui.clipboard.write(self.captions_service.transcript())
            ui.notify('Skopiowano napisy.')

        def download_text():
            text = self.captions_service.transcript()
            ui.download((text + '\n').encode('utf-8'), filename='napisy-na-zywo.txt', media_type='text/plain; charset=utf-8')

        start.on_click(start_live)
        stop.on_click(stop_live)
        copy.on_click(copy_text)
        download.on_click(download_text)

        def refresh():
            current = self.captions_service.snapshot()
            active = current['phase'] in ('preparing', 'preparing_speakers', 'waiting_gpu', 'listening', 'stopping')
            own = current['owner'] == self.owner_id
            messages = {'idle': 'Gotowość', 'preparing': 'Ładowanie modelu…',
                        'preparing_speakers': 'Przygotowywanie modeli rozmówców…',
                        'waiting_gpu': 'Oczekiwanie na dostęp do GPU…', 'listening': 'Mikrofon włączony · napisy na żywo',
                        'stopping': 'Zatrzymywanie mikrofonu · kończenie napisów…', 'stopped': 'Zatrzymano', 'error': 'Błąd'}
            message = messages.get(current['phase'], current['phase'])
            if active and not own:
                message = 'Sesja napisów jest uruchomiona w innej karcie.'
            if current['error']:
                message = f"Błąd: {current['error']}"
            status.set_text(message)
            status.classes(add='live-microphone-active' if current['phase'] == 'listening' else '',
                           remove='' if current['phase'] == 'listening' else 'live-microphone-active')
            level.value = current['level']
            start.set_enabled(bool(devices) and not active)
            stop.set_enabled(active and own and current['phase'] != 'stopping')
            copy.set_enabled(bool(current['line_count']))
            download.set_enabled(bool(current['line_count']))
            update_controls(active)
            notes = []
            if current['pending']:
                notes.append(f"Oczekujące porcje: {current['pending']}")
            if current['dropped_seconds']:
                notes.append(f"Pominięto około {current['dropped_seconds']:.1f} s dźwięku — wybierz mniejszy model lub sprawdź obciążenie komputera.")
            if current['input_warnings']:
                notes.append('Mikrofon zgłosił zakłócenia wejścia; część dźwięku mogła zostać utracona.')
            warning.set_text(' · '.join(notes))
            if current['revision'] != last_revision['value']:
                last_revision['value'] = current['revision']
                output.set_text(current['text'] or 'Tutaj pojawią się napisy.')
                if current['text']:
                    scroll.scroll_to(percent=1)

        device.on_value_change(lambda _: refresh())
        refresh()
        self.timer(.4, refresh)

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
                                    if job['status'] != 'waiting':
                                        info += f" · Czas od uruchomienia: {short_time(self.tasks.elapsed(job))}"
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
                                args = (options.model, options.cache_dir, 'vulkan') if options.device == 'vulkan' else (options.model, options.cache_dir)
                                detail = self.model_manager.details(*args)
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
            elapsed_time = ui.label('').classes('text-sm')
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
                elapsed_time.set_text(f"Czas od uruchomienia: {short_time(self.tasks.elapsed(current))}")
                eta.set_text(eta_text(current))
                result_button.set_visibility(current['status'] == 'done')
                rows = self.db.live_segments(job_id, after=last_sequence)
                if rows:
                    empty.set_visibility(False)
                    lines = []
                    for row in rows:
                        lines.append(f"[{short_time(row['start'])}] {segment_text(row)}")
                        texts.append(segment_text(row))
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
            model_format = ui.select({'cpu': 'Faster-Whisper · CPU/CUDA', 'vulkan': 'Whisper.cpp · Vulkan'},
                                     label='Format modeli', value='vulkan' if settings.device == 'vulkan' else 'cpu').classes('w-full').props('outlined dense')
            ui.label('Modele Vulkan są osobnymi plikami GGML; transkrypcja pobiera brakujący model automatycznie.').classes('hint')
            elements = {}
            for name in MODEL_REPOS:
                with ui.row().classes('w-full model-row items-center no-wrap') as model_row:
                    with ui.column().classes('grow gap-0'):
                        ui.label(name).classes('font-medium text-base')
                        info = ui.label('').classes('model-info')
                        bar = ui.linear_progress(value=0, size='7px', show_value=False, color='light-blue-5').classes('w-full q-mt-xs').props('rounded')
                    with ui.row().classes('model-badge items-center no-wrap') as badge:
                        badge_icon = ui.icon('cloud_download', size='17px')
                        badge_text = ui.label('NIEPOBRANY')
                    download = ui.button('Pobierz', icon='download', on_click=lambda _, n=name: self.models_start(n, settings, model_format.value)).props('flat no-caps')
                    def cancel(_, n=name):
                        self.models_cancel(n, settings, model_format.value)
                        refresh()
                    cancel_button = ui.button('Anuluj', icon='cancel', on_click=cancel, color='negative').props('flat no-caps')
                    async def remove(n=name):
                        try:
                            selected_format = model_format.value
                            active = self.db.list(('downloading', 'preparing', 'transcribing', 'exporting'))
                            if any(Settings.model_validate_json(j['options']).model == n for j in active):
                                raise RuntimeError('Model jest używany przez aktywne zadanie.')
                            if self.captions_service.using_model(n, settings.cache_dir):
                                raise RuntimeError('Model jest używany przez napisy na żywo.')
                            await run.io_bound(self.tasks.gpu_engine.unload)
                            args = (n, settings.cache_dir, 'vulkan') if selected_format == 'vulkan' else (n, settings.cache_dir)
                            await run.io_bound(self.model_manager.delete, *args)
                            ui.notify('Usunięto model.', type='positive')
                        except Exception as exc:
                            ui.notify(str(exc), type='negative')
                    delete = ui.button(icon='delete_outline', on_click=remove).props('flat round').tooltip('Usuń model')
                    elements[name] = (model_row, info, bar, badge, badge_icon, badge_text, download, cancel_button, delete)
            labels = {
                'missing': ('NIEPOBRANY', 'cloud_off', 'Pobierz', 'download'),
                'partial': ('NIEKOMPLETNY', 'cloud_sync', 'Wznów', 'play_arrow'),
                'checking': ('SPRAWDZANIE', 'sync', 'Sprawdzanie…', 'sync'),
                'downloading': ('POBIERANIE', 'sync', 'Pobieranie…', 'download'),
                'cancelling': ('ANULOWANIE', 'sync', 'Anulowanie…', 'sync'),
                'cancelled': ('ANULOWANO', 'cancel', 'Wznów', 'refresh'),
                'ready': ('POBRANY · GOTOWY OFFLINE', 'check_circle', 'Pobrano', 'check'),
                'error': ('BŁĄD POBIERANIA', 'error_outline', 'Ponów', 'refresh'),
            }

            def refresh():
                available_count = 0
                for name, (row, info, bar, badge, icon, badge_text, button, cancel_button, delete) in elements.items():
                    args = (name, settings.cache_dir, 'vulkan') if model_format.value == 'vulkan' else (name, settings.cache_dir)
                    detail = self.model_manager.details(*args)
                    state = model_state(detail)
                    available_count += int(state == 'ready')
                    running = state in ('checking', 'downloading', 'cancelling')
                    color = 'downloading' if running else 'partial' if state == 'cancelled' else state
                    label, icon_name, button_text, button_icon = labels[state]
                    total = detail.get('total') or 0
                    downloaded = detail.get('downloaded') or 0
                    progress = min(1.0, max(0.0, downloaded / total)) if total else 0.0
                    info_text = f"Na dysku: {human_size(detail['size'])} · rozmiar modelu {detail['estimated']}"
                    if state == 'checking':
                        info_text += ' · Sprawdzanie plików…'
                    elif state == 'downloading':
                        if total:
                            percent = f'{progress * 100:.1f}'.replace('.', ',')
                            label += f' · {percent}%'
                            info_text += f" · Pobrano {human_size(downloaded)} z {human_size(total)}"
                        else:
                            info_text += ' · Pobieranie; całkowity rozmiar nie jest jeszcze znany'
                    elif state == 'partial':
                        info_text += ' · Pobieranie nie zostało ukończone'
                    elif state == 'cancelling':
                        info_text += ' · Trwa zatrzymywanie pobierania…'
                    elif state == 'cancelled':
                        info_text += ' · Pobieranie anulowane; możesz je wznowić'
                    elif state == 'error':
                        info_text += f" · {detail['status']}"
                    info.set_text(info_text)
                    other_colors = ' '.join(item for item in ('missing', 'partial', 'downloading', 'ready', 'error') if item != color)
                    row.classes(add=color, remove=other_colors)
                    badge.classes(add=color, remove=other_colors)
                    icon.set_name(icon_name)
                    icon.classes(add='model-spin' if running else '', remove='' if running else 'model-spin')
                    badge_text.set_text(label)
                    bar.value = progress
                    bar.props(add='indeterminate' if running and not total else '', remove='' if running and not total else 'indeterminate')
                    bar.set_visibility(running)
                    button.set_text(button_text)
                    button.props(f'icon={button_icon}')
                    button.set_enabled(not running and state != 'ready')
                    cancel_button.set_visibility(running)
                    cancel_button.set_enabled(state != 'cancelling')
                    delete.set_enabled(bool(detail['size']) and not running)
                summary.set_text(f'Pobrane: {available_count} z {len(elements)} modeli')
            refresh()
            model_format.on_value_change(lambda _: refresh())
            self.timer(1.5, refresh)

    def models_start(self, name, settings, device='cpu'):
        args = (name, settings.cache_dir, 'vulkan') if device == 'vulkan' else (name, settings.cache_dir)
        self.model_manager.start_download(*args)
        ui.notify(f'Pobieranie modelu {name} rozpoczęte.')

    def models_cancel(self, name, settings, device='cpu'):
        args = (name, settings.cache_dir, 'vulkan') if device == 'vulkan' else (name, settings.cache_dir)
        if self.model_manager.cancel_download(*args):
            ui.notify(f'Anulowanie pobierania modelu {name}…')
        else:
            ui.notify('To pobieranie już się zakończyło.')

    def editor(self, job_id):
        job = self.db.get(job_id)
        if not job or not job['result']:
            header('Nie znaleziono wyniku', 'Wybierz ukończoną transkrypcję z historii.')
            return
        result = json.loads(job['result'])
        header('Edytor transkrypcji', job['name'])
        with ui.row().classes('w-full gap-3 items-center q-mb-md'):
            device_text = result.get('device', '—').upper()
            if result.get('gpu_name'):
                device_text += f" {result.get('gpu_index')} · {result['gpu_name']}"
            ui.label(f"{result.get('model', '—')} · {device_text} · {result.get('language', '—')} · RTF {result.get('rtf', 0) or 0:.2f}").classes('hint grow')
            ui.label(f"Czas od uruchomienia: {short_time(job.get('elapsed'))}").classes('hint')
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
            media_path = Path(job['path'])
            if media_path.is_file():
                player = (ui.video(f'/media/{job_id}') if media_path.suffix.lower() in VIDEO_EXTENSIONS
                          else ui.audio(f'/media/{job_id}')).classes('w-full')
            else:
                player = None
                ui.label('Nie znaleziono pliku nagrania w zapisanej lokalizacji. Transkrypcję można nadal edytować i eksportować.').classes('text-warning')
            with ui.row().classes('w-full items-center'):
                search = ui.input(placeholder='Szukaj w transkrypcji…').props('outlined dense clearable').classes('grow')
                def copy():
                    current = json.loads(self.db.get(job_id)['result'])
                    ui.clipboard.write('\n'.join(segment_text(s) for s in current['segments']))
                    ui.notify('Skopiowano transkrypcję.')
                ui.button('Kopiuj całość', icon='content_copy', on_click=copy).props('flat no-caps')
            ui.label('Zmiany zapisują się automatycznie. Kliknij znacznik czasu, aby przejść do fragmentu.').classes('hint')
            speaker_filter = ui.select(speaker_filter_options(result['segments']), label='Filtr rozmówców',
                                       value=[], multiple=True).classes('w-full').props('outlined dense clearable use-chips')
            ui.label('Bez zaznaczenia widoczni są wszyscy rozmówcy. Filtr obejmuje również nieustalone głosy.').classes('hint')
            with ui.row().classes('w-full items-center'):
                speaker_edit = ui.switch('Edytuj rozmówców', value=bool(result.get('diarization') or any(s.get('speaker') for s in result['segments'])))
                show_timestamps = ui.switch('Pokaż znaczniki czasu', value=True)
            state = {'page': 1}

            def change_page(event):
                page = int(event.value or 1)
                if page != state['page']:
                    state['page'] = page
                    draw()

            with ui.row().classes('w-full items-center justify-between'):
                summary = ui.label('').classes('hint')
                top_pager = ui.pagination(1, 1, value=1, direction_links=True, on_change=change_page).props('max-pages=7 boundary-numbers')
            rows = ui.column().classes('w-full gap-0')
            bottom_pager = ui.pagination(1, 1, value=1, direction_links=True, on_change=change_page).props('max-pages=7 boundary-numbers')

            def draw():
                rows.clear()
                current = json.loads(self.db.get(job_id)['result'])
                matches = matching_segments(current['segments'], search.value or '', speaker_filter.value)
                pages = max(1, (len(matches) + EDITOR_PAGE_SIZE - 1) // EDITOR_PAGE_SIZE)
                state['page'] = min(max(1, state['page']), pages)
                start = (state['page'] - 1) * EDITOR_PAGE_SIZE
                end = min(start + EDITOR_PAGE_SIZE, len(matches))
                summary.set_text(f'Fragmenty {start + 1}–{end} z {len(matches)}' if matches else 'Brak pasujących fragmentów.')
                for pager in (top_pager, bottom_pager):
                    pager.max = pages
                    if pager.value != state['page']:
                        pager.value = state['page']
                    pager.set_visibility(pages > 1)
                with rows:
                    for index, segment in matches[start:end]:
                        with ui.row().classes('w-full job-row no-wrap items-start'):
                            def seek(second=segment['start']):
                                if player:
                                    player.seek(float(second))
                                    player.play()
                            if show_timestamps.value:
                                seek_button = ui.button(short_time(segment['start']), on_click=seek).props('flat no-caps dense').classes('text-primary')
                                seek_button.set_enabled(player is not None)
                            if speaker_edit.value:
                                def save_speaker(e, idx=index):
                                    try:
                                        self.db.edit_speaker(job_id, idx, e.value or '')
                                        updated = json.loads(self.db.get(job_id)['result'])
                                        speaker_filter.set_options(speaker_filter_options(updated['segments'], speaker_filter.value or []))
                                    except Exception as exc:
                                        ui.notify(str(exc), type='negative')
                                speaker_field = ui.input(value=segment.get('speaker') or '', placeholder='Nieustalony',
                                                         on_change=save_speaker).classes('w-40').props('outlined dense maxlength=80')
                                speaker_field.on('blur', lambda: draw() if speaker_filter.value else None)
                            elif segment.get('speaker'):
                                ui.label(segment['speaker']).classes('hint w-40')
                            def save(e, idx=index):
                                try:
                                    self.db.edit_segment(job_id, idx, e.value or '')
                                except Exception as exc:
                                    ui.notify(str(exc), type='negative')
                            ui.textarea(value=segment['text'], on_change=save).classes('grow').props('outlined autogrow')
                            if show_timestamps.value and segment.get('words'):
                                ui.label(' '.join(f"{w['word'].strip()} [{short_time(w['start'])}]" for w in segment['words'])).classes('hint')

            def search_changed(_):
                state['page'] = 1
                draw()

            search.on_value_change(search_changed)
            speaker_filter.on_value_change(search_changed)
            speaker_edit.on_value_change(lambda _: draw())
            show_timestamps.on_value_change(lambda _: draw())
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
            gpu, _ = gpu_picker(self.db.settings().gpu_id)
            vulkan_gpu = ui.number('Numer GPU Vulkan do testu', min=0, max=31, step=1, precision=0,
                                   value=self.db.settings().vulkan_device).props('outlined')
            summary = ui.label('Odczytywanie informacji…').classes('whitespace-pre-wrap text-sm w-full break-all')
            revision = {'value': 0}
            async def refresh():
                revision['value'] += 1
                request = revision['value']
                data = await run.io_bound(diagnostics, gpu.value or '')
                if not summary.is_deleted and revision['value'] == request:
                    summary.set_text(json.dumps(data, ensure_ascii=False, indent=2))
            async def test_gpu():
                revision['value'] += 1
                request = revision['value']
                summary.set_text('Trwa rzeczywisty test Faster-Whisper na CUDA…')
                settings = self.db.settings()
                path = self.model_manager.path('tiny', settings.cache_dir)
                if not self.model_manager.available('tiny', settings.cache_dir):
                    summary.set_text('Brak modelu tiny. Pobierz go w zakładce Modele AI, a następnie ponów test.')
                    return
                result = await run.io_bound(inference_test, path, gpu_id=gpu.value or '')
                if not summary.is_deleted and revision['value'] == request:
                    summary.set_text(json.dumps(result, ensure_ascii=False, indent=2))
                    ui.notify('Test GPU zakończony.' if result.get('ok') else 'Test GPU nie powiódł się. Sprawdź szczegóły.', type='positive' if result.get('ok') else 'negative')

            async def test_vulkan():
                revision['value'] += 1
                request = revision['value']
                settings = self.db.settings()
                if not self.model_manager.available('tiny', settings.cache_dir, 'vulkan'):
                    summary.set_text('Brak modelu tiny dla Vulkan. Pobierz go w zakładce Modele AI i ponów test.')
                    return
                if not self.tasks.gpu_slot.acquire(blocking=False):
                    summary.set_text('GPU jest zajęte. Zakończ inne zadanie lub sesję napisów i ponów test.')
                    return
                summary.set_text('Trwa rzeczywisty test whisper.cpp na Vulkan…')
                try:
                    from app.config import ROOT
                    options = Settings(model='tiny', device='vulkan', cache_dir=settings.cache_dir,
                                       vulkan_device=int(vulkan_gpu.value or 0), vad=False, beam_size=1)
                    result = await run.io_bound(self.tasks.vulkan_engine.transcribe,
                                                ROOT / 'tests' / 'fixtures' / 'polish.wav', options,
                                                lambda _: None, lambda: False)
                    data = {'ok': True, 'device': result['device'], 'gpu_index': result['gpu_index'],
                            'language': result['language'], 'segments': result['segments'][:3]}
                except Exception as exc:
                    data = {'ok': False, 'error': str(exc)}
                finally:
                    self.tasks.gpu_slot.release()
                if not summary.is_deleted and revision['value'] == request:
                    summary.set_text(json.dumps(data, ensure_ascii=False, indent=2))
                    ui.notify('Test Vulkan zakończony.' if data['ok'] else 'Test Vulkan nie powiódł się. Sprawdź szczegóły.',
                              type='positive' if data['ok'] else 'negative')
            with ui.row():
                ui.button('Odśwież', icon='refresh', on_click=refresh).props('outline no-caps')
                ui.button('Uruchom test CUDA', icon='play_arrow', on_click=test_gpu).props('unelevated no-caps')
                ui.button('Uruchom test Vulkan', icon='play_arrow', on_click=test_vulkan).props('unelevated no-caps')
            gpu.on_value_change(lambda _: refresh())
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
