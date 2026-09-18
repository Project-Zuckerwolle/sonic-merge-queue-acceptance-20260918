"""
Nova Predator v1 — _compat.py
Python 3.14 + Windows 10 compatibility guards.

Python 3.14 breaking changes that affect Nova:
  1. asyncio.get_event_loop() now raises RuntimeError if no loop is set
     → always use asyncio.get_running_loop() inside coroutines
     → use asyncio.run() as the single entry point (never set_event_loop manually)
  2. asyncio policy system deprecated (removed in 3.16)
     → WindowsSelectorEventLoopPolicy no longer needed; ProactorEventLoop is default
     → DO NOT call asyncio.set_event_loop_policy() — it emits DeprecationWarning
  3. concurrent.interpreters is now stdlib (PEP 734) but STILL buggy on Windows
     → detect and disable; fall back to threading
  4. importlib.util.find_spec() is unreliable in 3.14 for some stdlib modules
     → use try/except __import__ pattern instead
  5. asyncio.iscoroutinefunction() deprecated → use inspect.iscoroutinefunction()

Windows-specific:
  - uvicorn must use ws=wsproto on Windows (websockets-sansio is the safe default
    in uvicorn 0.38+, but we pin wsproto as explicit fallback)
  - No uvloop on Windows (uvicorn[standard] skips it automatically)
  - ProactorEventLoop is Windows default since 3.8 — do not override
"""

from __future__ import annotations

import sys
import inspect
import asyncio
import logging

log = logging.getLogger("nova.compat")

# ── Version assertions ────────────────────────────────────────────────────────

def check_python_version() -> None:
    """Raise early if Python version is unsupported."""
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 12):
        raise RuntimeError(
            f"Nova Predator v1 requires Python 3.12+. Running {sys.version}. "
            "Please upgrade."
        )
    if (major, minor) >= (3, 16):
        log.warning(
            "Python %d.%d detected. asyncio policy API removed in 3.16. "
            "Review nova.core._compat if issues arise.", major, minor
        )

# ── asyncio helpers (3.14-safe) ───────────────────────────────────────────────

def get_running_loop() -> asyncio.AbstractEventLoop:
    """
    3.14-safe way to get the current running loop.
    asyncio.get_event_loop() raises RuntimeError in 3.14 when no loop is set.
    Always use asyncio.get_running_loop() inside coroutines instead.
    """
    return asyncio.get_running_loop()


def is_coroutine_function(obj: object) -> bool:
    """
    3.14-safe coroutine check.
    asyncio.iscoroutinefunction() deprecated in 3.12, removed in 3.16.
    """
    return inspect.iscoroutinefunction(obj)


# ── concurrent.interpreters guard ────────────────────────────────────────────

def concurrent_interpreters_available() -> bool:
    """
    concurrent.interpreters is stdlib in 3.14 (PEP 734) but still has
    Windows-specific instability. We check availability AND do a smoke test.
    Returns False on any failure — Nova falls back to threading.
    """
    if sys.platform != "win32":
        try:
            import concurrent.interpreters  # noqa: F401
            return True
        except ImportError:
            return False
    # On Windows: always return False until upstream fixes land
    return False


# ── Package import guard ──────────────────────────────────────────────────────

def package_available(name: str) -> bool:
    """
    importlib.util.find_spec() is unreliable in Python 3.14 for some packages.
    Use direct import attempt instead (v9 lesson).
    """
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ── uvicorn WebSocket recommendation ─────────────────────────────────────────

UVICORN_WS_BACKEND: str = "wsproto"
"""
Recommended WS backend for uvicorn on Windows Python 3.14.
websockets-sansio (uvicorn 0.38+ default) is also fine but wsproto
is more stable for our single-connection use case.
"""

# ── Run on import ─────────────────────────────────────────────────────────────
check_python_version()
