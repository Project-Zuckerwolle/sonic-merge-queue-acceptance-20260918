"""Test-Stubs für nicht-installierte Packages im Build-Container."""
import sys
import json
import io
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock


class _AsyncWriteFile:
    def __init__(self, path, encoding="utf-8"):
        self._path = Path(path)
        self._enc = encoding
        self._buf = io.StringIO()

    async def write(self, data):
        self._buf.write(data)

    async def read(self):
        return self._path.read_text(encoding=self._enc) if self._path.exists() else ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(self._buf.getvalue(), encoding=self._enc)


class _AsyncReadFile:
    def __init__(self, path, encoding="utf-8"):
        self._path = Path(path)
        self._enc = encoding

    async def read(self):
        return self._path.read_text(encoding=self._enc)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass


class _AiofilesOpen:
    def __init__(self, path, mode="r", encoding="utf-8"):
        self._path = path
        self._mode = mode
        self._enc = encoding

    def __await__(self):
        # Nicht awaitable — gibt direkt Objekt zurück
        return self._make_file().__await__()

    async def _make_file(self):
        if "w" in self._mode:
            return _AsyncWriteFile(self._path, self._enc)
        return _AsyncReadFile(self._path, self._enc)

    async def __aenter__(self):
        if "w" in self._mode:
            self._file = _AsyncWriteFile(self._path, self._enc)
        else:
            self._file = _AsyncReadFile(self._path, self._enc)
        return self._file

    async def __aexit__(self, *args):
        if hasattr(self._file, '__aexit__'):
            await self._file.__aexit__(*args)


def aiofiles_open(path, mode="r", encoding="utf-8"):
    return _AiofilesOpen(path, mode, encoding)


# Module injizieren
if 'aiofiles' not in sys.modules:
    stub = MagicMock()
    stub.open = aiofiles_open
    sys.modules['aiofiles'] = stub

if 'ollama' not in sys.modules:
    ollama_stub = MagicMock()
    ollama_stub.AsyncClient = AsyncMock
    ollama_stub.ResponseError = Exception
    sys.modules['ollama'] = ollama_stub
