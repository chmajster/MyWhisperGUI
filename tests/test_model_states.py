from nicegui import ui
from nicegui.events import GenericEventArguments

from app.config import Settings
from app.core.model_manager import ModelManager
from app.ui.layout import Interface


def test_model_panel_distinguishes_states_and_updates_after_download(tmp_path, monkeypatch):
    states = {
        'tiny': {'available': False, 'size': 0},
        'base': {'available': False, 'size': 100},
        'small': {'available': False, 'size': 0, 'status': 'Sprawdzanie plików'},
        # Complete-looking files do not mean that an active download has finished.
        'medium': {'available': True, 'size': 40, 'status': 'Pobieranie', 'downloaded': 40, 'total': 100},
        'large-v3': {'available': True, 'size': 100, 'status': 'Pobrano'},
        'turbo': {'available': True, 'size': 20, 'status': 'Błąd: brak połączenia'},
    }

    class FakeDatabase:
        def settings(self):
            return Settings(cache_dir=str(tmp_path))

    class FakeModels:
        def details(self, name, cache_dir):
            return {'name': name, 'estimated': 'ok. 100 MB', **states[name]}

        def cancel_download(self, name, cache_dir):
            assert name == 'medium' and cache_dir == str(tmp_path)
            states[name]['status'] = 'Anulowanie'
            return True

    monkeypatch.setattr(ui, 'notify', lambda *args, **kwargs: None)
    interface = Interface.__new__(Interface)
    interface.db = FakeDatabase()
    interface.model_manager = FakeModels()
    callbacks = []
    interface.timer = lambda _, callback: callbacks.append(callback)

    with ui.card() as panel:
        interface.models()
    try:
        def row(name):
            return next(item for item in panel.descendants() if 'model-row' in item._classes
                        and any(getattr(child, 'text', None) == name for child in item.descendants()))

        def badge(model_row):
            return next(item for item in model_row.descendants() if 'model-badge' in item._classes)

        def button(model_row, label):
            return next(item for item in model_row.descendants() if item._props.get('label') == label)

        def progress(model_row):
            return next(item for item in model_row.descendants() if item.tag == 'q-linear-progress')

        expected = {
            'tiny': ('missing', 'NIEPOBRANY', 'Pobierz', True),
            'base': ('partial', 'NIEKOMPLETNY', 'Wznów', True),
            'small': ('downloading', 'SPRAWDZANIE', 'Sprawdzanie…', False),
            'medium': ('downloading', 'POBIERANIE · 40,0%', 'Pobieranie…', False),
            'large-v3': ('ready', 'POBRANY · GOTOWY OFFLINE', 'Pobrano', False),
            'turbo': ('error', 'BŁĄD POBIERANIA', 'Ponów', True),
        }
        for name, (color, label, action, enabled) in expected.items():
            model_row = row(name)
            assert color in model_row._classes and color in badge(model_row)._classes
            assert any(getattr(item, 'text', None) == label for item in model_row.descendants())
            assert button(model_row, action).enabled is enabled

        assert progress(row('small')).visible and 'indeterminate' in progress(row('small'))._props
        assert progress(row('medium')).visible and progress(row('medium')).value == .4
        assert not progress(row('large-v3')).visible
        assert not button(row('tiny'), 'Anuluj').visible
        assert button(row('small'), 'Anuluj').visible and button(row('small'), 'Anuluj').enabled
        assert button(row('medium'), 'Anuluj').visible and button(row('medium'), 'Anuluj').enabled
        assert not button(row('large-v3'), 'Anuluj').visible
        assert any(getattr(item, 'text', None) == 'Pobrane: 1 z 6 modeli' for item in panel.descendants())

        cancel_button = button(row('medium'), 'Anuluj')
        click = next(listener for listener in cancel_button._event_listeners.values() if listener.type == 'click')
        click.handler(GenericEventArguments(sender=cancel_button, client=cancel_button.client, args=[]))
        assert states['medium']['status'] == 'Anulowanie'
        assert cancel_button.visible and not cancel_button.enabled
        assert not button(row('medium'), 'Anulowanie…').enabled
        states['medium']['status'] = 'Anulowano'
        callbacks[0]()
        assert not cancel_button.visible
        assert 'partial' in badge(row('medium'))._classes
        assert button(row('medium'), 'Wznów').enabled
        assert any(getattr(item, 'text', None) == 'ANULOWANO' for item in row('medium').descendants())

        states['medium'] = {'available': True, 'size': 100, 'status': 'Pobrano'}
        callbacks[0]()
        model_row = row('medium')
        assert 'ready' in model_row._classes and 'downloading' not in model_row._classes
        assert 'ready' in badge(model_row)._classes and 'downloading' not in badge(model_row)._classes
        assert not progress(model_row).visible
        assert not button(model_row, 'Pobrano').enabled
        assert any(getattr(item, 'text', None) == 'Pobrane: 2 z 6 modeli' for item in panel.descendants())
    finally:
        panel.delete()


def test_model_details_survive_a_download_file_being_renamed(tmp_path, monkeypatch):
    directory = tmp_path / 'tiny'
    directory.mkdir()
    partial = directory / 'download.incomplete'
    partial.write_bytes(b'data')
    original_stat = type(partial).stat
    calls = 0

    def changing_stat(path, *args, **kwargs):
        nonlocal calls
        if path == partial:
            calls += 1
            if calls == 2:
                raise FileNotFoundError('The downloader moved this file')
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(type(partial), 'stat', changing_stat)
    detail = ModelManager().details('tiny', tmp_path)
    assert detail['available'] is False
    assert detail['size'] == 0
