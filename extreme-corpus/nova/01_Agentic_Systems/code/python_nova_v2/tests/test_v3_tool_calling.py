"""Tests für v3 Tool-Calling Infrastruktur.

Testet die neue Architektur die SemanticRouter + Pipeline ersetzt hat:
  - OllamaClient.chat_mit_tools() + Parsing
  - SkillMesh.alle_tool_definitionen() + tool_aufrufen()
  - memory_extraktion.extrahiere_und_speichere()

Keine echten Ollama-Calls — alles gemockt via AsyncMock.
"""
import pytest
import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Stubs laden (aiofiles, ollama) — muss VOR Nova-Imports passieren
from tests import _stubs  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════
# OllamaClient Tool-Parsing
# ═══════════════════════════════════════════════════════════════════════════

def test_tool_parse_einfach():
    """Modell antwortet mit einem Tool-Call — wird erkannt."""
    from core.ollama_client import _parse_tool_aufrufe
    antwort = '<tool_call>{"name": "wetter", "argumente": {"stadt": "Berlin"}}</tool_call>'
    calls, rest = _parse_tool_aufrufe(antwort)
    assert len(calls) == 1
    assert calls[0].name == "wetter"
    assert calls[0].argumente == {"stadt": "Berlin"}
    assert rest == ""


def test_tool_parse_mit_begleittext():
    """Text neben Tool-Call wird sauber getrennt."""
    from core.ollama_client import _parse_tool_aufrufe
    antwort = 'Ich prüfe das. <tool_call>{"name": "news", "argumente": {}}</tool_call>'
    calls, rest = _parse_tool_aufrufe(antwort)
    assert len(calls) == 1
    assert calls[0].name == "news"
    assert "Ich prüfe" in rest
    # Tag selbst nicht mehr im Rest
    assert "<tool_call>" not in rest


def test_tool_parse_keine_tools():
    """Normale Antwort ohne Tags — leere Liste, voller Text."""
    from core.ollama_client import _parse_tool_aufrufe
    antwort = "Das Wetter heute ist sonnig in Berlin."
    calls, rest = _parse_tool_aufrufe(antwort)
    assert calls == []
    assert rest == antwort


def test_tool_parse_mehrere_calls():
    """Mehrere Tool-Calls hintereinander werden alle erkannt."""
    from core.ollama_client import _parse_tool_aufrufe
    antwort = (
        '<tool_call>{"name": "wetter", "argumente": {"stadt": "Berlin"}}</tool_call>'
        '<tool_call>{"name": "news", "argumente": {"kategorie": "tech"}}</tool_call>'
    )
    calls, _ = _parse_tool_aufrufe(antwort)
    assert len(calls) == 2
    assert calls[0].name == "wetter"
    assert calls[1].name == "news"


def test_tool_parse_englische_argumente_keyword():
    """Akzeptiert sowohl 'argumente' (de) als auch 'arguments' (en)."""
    from core.ollama_client import _parse_tool_aufrufe
    antwort = '<tool_call>{"name": "wetter", "arguments": {"stadt": "Wien"}}</tool_call>'
    calls, _ = _parse_tool_aufrufe(antwort)
    assert len(calls) == 1
    assert calls[0].argumente == {"stadt": "Wien"}


def test_tool_parse_kaputtes_json_wird_ignoriert():
    """Ungültiges JSON wird stillschweigend verworfen — kein Crash."""
    from core.ollama_client import _parse_tool_aufrufe
    antwort = '<tool_call>{das ist kein json}</tool_call> aber das hier ist text'
    calls, rest = _parse_tool_aufrufe(antwort)
    assert calls == []
    # Tag wurde trotzdem entfernt
    assert "<tool_call>" not in rest


def test_tool_system_prompt_enthält_tool_namen():
    """Der generierte System-Prompt-Abschnitt listet Tools mit Beschreibungen."""
    from core.ollama_client import _tool_system_prompt
    tools = [
        {
            "type": "function",
            "function": {
                "name": "wetter",
                "description": "Wetterdaten abrufen",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stadt": {"type": "string", "description": "Stadtname"},
                    },
                    "required": ["stadt"],
                },
            },
        }
    ]
    prompt = _tool_system_prompt(tools)
    assert "wetter" in prompt
    assert "Wetterdaten abrufen" in prompt
    assert "stadt" in prompt
    assert "<tool_call>" in prompt  # Format-Anweisung enthalten


def test_tool_system_prompt_leer_bei_keinen_tools():
    """Keine Tools → leerer Prompt-Abschnitt."""
    from core.ollama_client import _tool_system_prompt
    assert _tool_system_prompt([]) == ""


# ═══════════════════════════════════════════════════════════════════════════
# OllamaClient.chat_mit_tools (mit Mock)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_chat_mit_tools_erkennt_tool_call():
    """chat_mit_tools gibt Tool-Aufrufe zurück wenn Modell einen Tag sendet."""
    from core.ollama_client import OllamaClient

    client = OllamaClient()
    # chat() direkt mocken — simuliert Ollama-Antwort
    client.chat = AsyncMock(return_value=(
        '<tool_call>{"name": "wetter", "argumente": {"stadt": "Berlin"}}</tool_call>'
    ))

    text, calls = await client.chat_mit_tools(
        nachrichten=[{"role": "user", "content": "wetter in Berlin"}],
        modell="gemma4:e4b",
        system="Du bist Nova.",
        tools=[{"type": "function", "function": {
            "name": "wetter", "description": "Wetter",
            "parameters": {"type": "object", "properties": {
                "stadt": {"type": "string", "description": "Stadt"}
            }, "required": ["stadt"]}
        }}],
    )

    assert len(calls) == 1
    assert calls[0].name == "wetter"
    assert calls[0].argumente == {"stadt": "Berlin"}


@pytest.mark.asyncio
async def test_chat_mit_tools_normale_antwort():
    """Ohne Tool-Tag → normale Antwort, leere Call-Liste."""
    from core.ollama_client import OllamaClient

    client = OllamaClient()
    client.chat = AsyncMock(return_value="Hallo, wie kann ich helfen?")

    text, calls = await client.chat_mit_tools(
        nachrichten=[{"role": "user", "content": "hallo"}],
        modell="gemma4:e4b",
        system="Du bist Nova.",
        tools=[],
    )

    assert calls == []
    assert "Hallo" in text


# ═══════════════════════════════════════════════════════════════════════════
# Skill tool_definition()
# ═══════════════════════════════════════════════════════════════════════════

def _lade_skill_modul(skill_name: str):
    """Lädt ein Skill-Modul dynamisch (wie der echte SkillLoader)."""
    import importlib.util
    modul_name = f"nova_skill_{skill_name}_test"
    spec = importlib.util.spec_from_file_location(
        modul_name, f"skills/{skill_name}/skill.py"
    )
    modul = importlib.util.module_from_spec(spec)
    sys.modules[modul_name] = modul  # nötig für @dataclass in skill-Modulen
    spec.loader.exec_module(modul)
    return modul


def test_alle_skills_haben_tool_definition():
    """Jeder der 4 Skills exportiert tool_definition()."""
    for name in ["wetter", "websearch", "news", "finanzen"]:
        modul = _lade_skill_modul(name)
        assert hasattr(modul, "tool_definition"), f"{name}: tool_definition() fehlt"
        td = modul.tool_definition()
        assert "function" in td, f"{name}: kein 'function' im Schema"
        fn = td["function"]
        assert fn.get("name") == name, f"{name}: Name im Schema stimmt nicht"
        assert "description" in fn, f"{name}: keine Beschreibung"
        assert "parameters" in fn, f"{name}: keine Parameter-Spec"


def test_wetter_tool_definition_struktur():
    """wetter hat 'stadt' als erforderlichen Parameter."""
    modul = _lade_skill_modul("wetter")
    td = modul.tool_definition()
    params = td["function"]["parameters"]
    assert "stadt" in params["properties"]
    assert "stadt" in params["required"]


# ═══════════════════════════════════════════════════════════════════════════
# SkillMesh Tool-API
# ═══════════════════════════════════════════════════════════════════════════

def _bau_registry_mit_dummy_skill():
    """Erstellt eine SkillRegistry mit einem synthetischen Test-Skill."""
    from core.skill_registry import (
        SkillRegistry, SkillRegistrierung, Hook, SkillResult
    )

    # Dummy-Modul mit tool_definition() und on_message()
    dummy_modul = MagicMock()
    dummy_modul.tool_definition = MagicMock(return_value={
        "type": "function",
        "function": {
            "name": "dummy",
            "description": "Test-Tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "wert": {"type": "string", "description": "Ein Wert"}
                },
                "required": ["wert"],
            },
        },
    })

    def dummy_fn(ctx):
        return SkillResult(inhalt=f"Dummy-Ergebnis: {ctx.skill_config.get('wert', '?')}")

    # Modul in sys.modules damit alle_tool_definitionen() es findet
    sys.modules["nova_skill_dummy_test"] = dummy_modul
    dummy_fn.__module__ = "nova_skill_dummy_test"

    registry = SkillRegistry()
    registry.registriere(SkillRegistrierung(
        name="dummy",
        hook=Hook.ON_MESSAGE,
        threshold=0.72,
        funktion=dummy_fn,
        config={},
    ))
    return registry


def test_skill_mesh_alle_tool_definitionen():
    """alle_tool_definitionen() liefert Schemas für alle registrierten Skills."""
    from orchestrator.skill_mesh import SkillMesh

    registry = _bau_registry_mit_dummy_skill()
    mesh = SkillMesh(registry)

    tools = mesh.alle_tool_definitionen()
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "dummy"


@pytest.mark.asyncio
async def test_skill_mesh_tool_aufrufen():
    """tool_aufrufen() führt Skill aus und gibt inhalt-String zurück."""
    from orchestrator.skill_mesh import SkillMesh

    registry = _bau_registry_mit_dummy_skill()
    mesh = SkillMesh(registry)

    ergebnis = await mesh.tool_aufrufen(
        tool_name="dummy",
        argumente={"wert": "test123"},
        user_input="egal",
    )

    assert isinstance(ergebnis, str)
    assert "test123" in ergebnis


@pytest.mark.asyncio
async def test_skill_mesh_tool_aufrufen_unbekannt():
    """Unbekanntes Tool gibt Fehlermeldung zurück — kein Crash."""
    from orchestrator.skill_mesh import SkillMesh

    registry = _bau_registry_mit_dummy_skill()
    mesh = SkillMesh(registry)

    ergebnis = await mesh.tool_aufrufen(
        tool_name="gibts_nicht",
        argumente={},
    )

    assert isinstance(ergebnis, str)
    assert "nicht verfügbar" in ergebnis.lower() or "not found" in ergebnis.lower() \
           or "gibts_nicht" in ergebnis


@pytest.mark.asyncio
async def test_skill_mesh_tool_aufrufen_skill_wirft_exception():
    """Wenn ein Skill crasht wird eine Fehlermeldung als String zurückgegeben."""
    from core.skill_registry import (
        SkillRegistry, SkillRegistrierung, Hook
    )
    from orchestrator.skill_mesh import SkillMesh

    def crasher(ctx):
        raise RuntimeError("boom")
    crasher.__module__ = "nova_skill_crash_test"

    registry = SkillRegistry()
    registry.registriere(SkillRegistrierung(
        name="crasher",
        hook=Hook.ON_MESSAGE,
        threshold=0.72,
        funktion=crasher,
        config={},
    ))
    mesh = SkillMesh(registry)

    ergebnis = await mesh.tool_aufrufen(tool_name="crasher", argumente={})
    assert "Fehler" in ergebnis
    assert "boom" in ergebnis


# ═══════════════════════════════════════════════════════════════════════════
# Memory-Extraktion
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_memory_extraktion_speichert_fakt():
    """LLM liefert JSON mit einem Fakt → wird ins Brain geschrieben."""
    from core.memory_extraktion import extrahiere_und_speichere
    from core.brain_manager import BrainManager
    from core.brain_index import BrainIndex

    with tempfile.TemporaryDirectory() as tmp:
        brain = BrainManager(brain_pfad=tmp)
        await brain.laden()
        index = BrainIndex()

        ollama = MagicMock()
        ollama.chat = AsyncMock(return_value='''[
            {"inhalt": "User arbeitet an Nova Predator v3", "typ": "fakt", "tags": ["nova", "projekt"]}
        ]''')
        ollama.embed = AsyncMock(return_value=[0.1] * 384)  # Dummy-Vektor

        gespeichert = await extrahiere_und_speichere(
            ollama=ollama,
            brain_manager=brain,
            brain_index=index,
            bm25_index=None,
            modell="gemma4:e4b",
            embed_modell="mxbai-embed-large",
            user_input="ich arbeite gerade an nova predator v3",
            antwort="Schön! Wie läuft es?",
        )

        assert len(gespeichert) == 1
        assert "Nova Predator v3" in gespeichert[0]

        # Brain hat jetzt einen Eintrag
        alle = await brain.alle()
        assert len(alle) == 1
        assert alle[0].typ == "fakt"


@pytest.mark.asyncio
async def test_memory_extraktion_ignoriert_kurzes_gespraech():
    """Zu kurze Turns werden ohne LLM-Call übersprungen."""
    from core.memory_extraktion import extrahiere_und_speichere
    from core.brain_manager import BrainManager
    from core.brain_index import BrainIndex

    with tempfile.TemporaryDirectory() as tmp:
        brain = BrainManager(brain_pfad=tmp)
        await brain.laden()
        index = BrainIndex()

        ollama = MagicMock()
        ollama.chat = AsyncMock(return_value="[]")
        ollama.embed = AsyncMock(return_value=[])

        gespeichert = await extrahiere_und_speichere(
            ollama=ollama,
            brain_manager=brain,
            brain_index=index,
            bm25_index=None,
            modell="gemma4:e4b",
            embed_modell="mxbai-embed-large",
            user_input="hi",
            antwort="hallo",
        )

        assert gespeichert == []
        # chat() wurde nicht aufgerufen — Vor-Filter hat gegriffen
        assert ollama.chat.call_count == 0


@pytest.mark.asyncio
async def test_memory_extraktion_leere_antwort():
    """Leere JSON-Antwort vom LLM → nichts gespeichert, kein Crash."""
    from core.memory_extraktion import extrahiere_und_speichere
    from core.brain_manager import BrainManager
    from core.brain_index import BrainIndex

    with tempfile.TemporaryDirectory() as tmp:
        brain = BrainManager(brain_pfad=tmp)
        await brain.laden()
        index = BrainIndex()

        ollama = MagicMock()
        ollama.chat = AsyncMock(return_value="[]")
        ollama.embed = AsyncMock(return_value=[])

        gespeichert = await extrahiere_und_speichere(
            ollama=ollama,
            brain_manager=brain,
            brain_index=index,
            bm25_index=None,
            modell="gemma4:e4b",
            embed_modell="mxbai-embed-large",
            user_input="was ist die hauptstadt von frankreich",
            antwort="Die Hauptstadt von Frankreich ist Paris.",
        )

        assert gespeichert == []


@pytest.mark.asyncio
async def test_memory_extraktion_kaputtes_json_kein_crash():
    """Wenn LLM kein valides JSON liefert, wird einfach nichts gespeichert."""
    from core.memory_extraktion import extrahiere_und_speichere
    from core.brain_manager import BrainManager
    from core.brain_index import BrainIndex

    with tempfile.TemporaryDirectory() as tmp:
        brain = BrainManager(brain_pfad=tmp)
        await brain.laden()
        index = BrainIndex()

        ollama = MagicMock()
        ollama.chat = AsyncMock(return_value="Hier ist nur Text kein JSON haha")
        ollama.embed = AsyncMock(return_value=[])

        gespeichert = await extrahiere_und_speichere(
            ollama=ollama,
            brain_manager=brain,
            brain_index=index,
            bm25_index=None,
            modell="gemma4:e4b",
            embed_modell="mxbai-embed-large",
            user_input="User-Input mit Länge über zehn Zeichen",
            antwort="Antwort mit Länge über zehn Zeichen.",
        )

        assert gespeichert == []


@pytest.mark.asyncio
async def test_memory_extraktion_sicher_fängt_exception():
    """extrahiere_sicher() propagiert niemals Fehler nach oben."""
    from core.memory_extraktion import extrahiere_sicher

    # Broken ollama — wirft bei jedem Aufruf
    ollama = MagicMock()
    ollama.chat = AsyncMock(side_effect=RuntimeError("ollama tot"))

    # Muss ohne Crash zurückkehren
    await extrahiere_sicher(
        ollama=ollama,
        brain_manager=MagicMock(),
        brain_index=MagicMock(),
        bm25_index=None,
        modell="gemma4:e4b",
        embed_modell="mxbai-embed-large",
        user_input="User-Input mit Länge über zehn Zeichen",
        antwort="Antwort mit Länge über zehn Zeichen.",
    )
    # Kein assert nötig — reiner Nicht-Crash-Test
