"""Exercise a clean Windows migration without changing the old environment."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.name != 'nt' or sys.version_info[:2] != (3, 12),
                    reason='The Windows installer test requires Python 3.12')
@pytest.mark.parametrize('source', ['explicit', 'uv'])
def test_installer_creates_python312_environment_and_preserves_previous_one(tmp_path, source):
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / 'install.ps1', tmp_path / 'install.ps1')
    (tmp_path / 'requirements.txt').write_text('', encoding='utf-8')
    module = tmp_path / 'app' / 'core'
    module.mkdir(parents=True)
    (module / 'gpu_manager.py').write_text('print("diagnostics ok")\n', encoding='utf-8')
    old_environment = tmp_path / '.venv'
    old_environment.mkdir()
    marker = old_environment / 'keep.txt'
    marker.write_text('previous environment', encoding='utf-8')
    environment = os.environ.copy()
    environment.update(PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    if source == 'explicit':
        environment['WHYSPER_PYTHON'] = sys.executable
    else:
        uv = shutil.which('uv')
        if not uv:
            pytest.skip('uv is not installed')
        environment.pop('WHYSPER_PYTHON', None)
        environment['PATH'] = os.pathsep.join((str(Path(uv).parent), str(Path(os.environ['SystemRoot']) / 'System32')))
        environment['UV_CACHE_DIR'] = str(tmp_path / 'uv-check-cache')
        available = subprocess.run([uv, 'python', 'find', '3.12', '--offline'], cwd=tmp_path,
                                   env=environment, capture_output=True, timeout=30)
        if available.returncode:
            pytest.skip('No uv-managed Python 3.12 is available')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                             str(tmp_path / 'install.ps1'), '-SkipCuda', '-SkipGpuTest'],
                            cwd=tmp_path, env=environment, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    python = tmp_path / '.venv-py312' / 'Scripts' / 'python.exe'
    assert subprocess.check_output([str(python), '--version'], text=True).startswith('Python 3.12.')
    assert marker.read_text(encoding='utf-8') == 'previous environment'
