"""Tests für Layer 1 — Memory-System."""
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


# ─── SessionFacts ─────────────────────────────────────────────────────────────

def test_session_facts_add_und_lesen():
    from memory.session_facts import SessionFacts
    sf = SessionFacts()
    sf.add("User nutzt Python 3.14", "fakt", 0.95)
    alle = sf.alle()
    assert len(alle) == 1
    assert alle[0].text == "User nutzt Python 3.14"

def test_session_facts_kein_duplikat():
    from memory.session_facts import SessionFacts
    sf = SessionFacts()
    sf.add("Python 3.14", "fakt")
    sf.add("Python 3.14", "fakt")  # Duplikat
    assert len(sf) == 1

def test_session_facts_reset():
    from memory.session_facts import SessionFacts
    sf = SessionFacts()
    sf.add("Test", "fakt")
    sf.reset()
    assert len(sf) == 0

def test_session_facts_kontext_text():
    from memory.session_facts import SessionFacts
    sf = SessionFacts()
    sf.add("User arbeitet an Nova", "fakt", 0.9)
    text = sf.als_kontext_text()
    assert "Nova" in text
    assert "Fakten" in text

def test_session_facts_max_fakten():
    from memory.session_facts import SessionFacts
    sf = SessionFacts(max_fakten=3)
    for i in range(5):
        sf.add(f"Fakt {i}", konfidenz=float(i) * 0.1 + 0.5)
    assert len(sf) <= 3

def test_session_facts_als_tuple_frozen():
    from memory.session_facts import SessionFacts, SessionFact
    sf = SessionFacts()
    sf.add("Test")
    t = sf.als_tuple()
    assert isinstance(t, tuple)
    assert isinstance(t[0], SessionFact)


# ─── WorkingMemory ───────────────────────────────────────────────────────────

def test_working_memory_reset():
    from memory.working_memory import WorkingMemory
    wm = WorkingMemory()
    wm.user_input = "hallo"
    wm.keywords = ["hallo"]
    wm.reset()
    assert wm.user_input == ""
    assert wm.keywords == []

def test_working_memory_als_dict():
    from memory.working_memory import WorkingMemory
    wm = WorkingMemory()
    wm.user_input = "Testfrage"
    wm.intents = ["code"]
    d = wm.als_dict()
    assert "user_input" in d
    assert "intents" in d


# ─── EpisodicMemory ──────────────────────────────────────────────────────────

def test_episodic_add_und_nachrichten():
    from memory.episodic_memory import EpisodicMemory
    em = EpisodicMemory(max_tokens=1000)
    em.add("user", "Hallo Nova")
    em.add("assistant", "Hallo! Was kann ich tun?")
    nachrichten = em.als_llm_nachrichten()
    assert len(nachrichten) == 2
    assert nachrichten[0]["role"] == "user"
    assert nachrichten[1]["role"] == "assistant"

def test_episodic_token_budget():
    from memory.episodic_memory import EpisodicMemory
    em = EpisodicMemory(max_tokens=50)  # Sehr klein
    for i in range(20):
        em.add("user", "x" * 20)    # Je ~5 Tokens
    nachrichten = em.als_llm_nachrichten()
    # Muss innerhalb Budget bleiben
    total = sum(len(m["content"]) // 4 for m in nachrichten)
    assert total <= 50 + 5    # kleiner Puffer

def test_episodic_kuerzen():
    from memory.episodic_memory import EpisodicMemory
    em = EpisodicMemory(max_tokens=100)
    for _ in range(10):
        em.add("user", "kurz")
        em.add("assistant", "antwort")
    vorher = len(em._verlauf)
    entfernt = em.kuerze(auf_prozent=0.4)
    assert entfernt > 0
    assert len(em._verlauf) < vorher

@pytest.mark.asyncio
async def test_episodic_archivieren():
    with tempfile.TemporaryDirectory() as tmp:
        from memory.episodic_memory import EpisodicMemory
        em = EpisodicMemory(sessions_pfad=tmp)
        em.add("user", "Archivierungstest")
        datei = await em.archiviere()
        assert datei.exists()
        assert len(em._verlauf) == 0   # Reset nach Archivierung

def test_episodic_status():
    from memory.episodic_memory import EpisodicMemory
    em = EpisodicMemory()
    em.add("user", "test")
    s = em.status()
    assert "turns" in s
    assert s["turns"] == 1


# ─── ContextBuilder ──────────────────────────────────────────────────────────

def test_context_builder_basis():
    from memory.context_builder import ContextBuilder
    from memory.episodic_memory import EpisodicMemory
    from memory.session_facts import SessionFacts
    import tempfile
    from core.persona import Persona

    with tempfile.TemporaryDirectory() as tmp:
        pf = Path(tmp) / "persona.yaml"
        pf.write_text("name: Nova\nton: test\nsprache: de\n", encoding="utf-8")
        persona = Persona(pfad=pf)
        episodic = EpisodicMemory()
        sf = SessionFacts()
        cb = ContextBuilder(persona=persona, episodic=episodic,
                            session_facts=sf, max_tokens=2000)
        ergebnis = cb.baue("Was ist Python?")
        assert ergebnis.system != ""
        assert len(ergebnis.messages) >= 1
        assert ergebnis.messages[-1]["role"] == "user"
        assert ergebnis.messages[-1]["content"] == "Was ist Python?"

def test_context_builder_mit_brain_fakten():
    from memory.context_builder import ContextBuilder
    from memory.episodic_memory import EpisodicMemory
    from memory.session_facts import SessionFacts
    import tempfile
    from core.persona import Persona

    with tempfile.TemporaryDirectory() as tmp:
        pf = Path(tmp) / "persona.yaml"
        pf.write_text("name: Nova\nton: test\nsprache: de\n", encoding="utf-8")
        persona = Persona(pfad=pf)
        cb = ContextBuilder(persona=persona, episodic=EpisodicMemory(),
                            session_facts=SessionFacts(), max_tokens=2000)
        ergebnis = cb.baue("Test", brain_fakten=["User liebt Python"])
        assert "Python" in ergebnis.system
        assert ergebnis.brain_fakten == ["User liebt Python"]

def test_context_builder_token_limit():
    from memory.context_builder import ContextBuilder
    from memory.episodic_memory import EpisodicMemory
    from memory.session_facts import SessionFacts
    import tempfile
    from core.persona import Persona

    with tempfile.TemporaryDirectory() as tmp:
        pf = Path(tmp) / "persona.yaml"
        pf.write_text("name: Nova\nton: test\nsprache: de\n", encoding="utf-8")
        persona = Persona(pfad=pf)
        episodic = EpisodicMemory(max_tokens=500)
        # Viel History laden
        for _ in range(50):
            episodic.add("user", "eine längere Frage die Platz braucht im Budget")
            episodic.add("assistant", "eine längere Antwort die ebenfalls Platz braucht")

        cb = ContextBuilder(persona=persona, episodic=episodic,
                            session_facts=SessionFacts(), max_tokens=500)
        ergebnis = cb.baue("kurze Frage")
        # Auslastung darf nicht massiv überschritten sein
        assert ergebnis.auslastung <= 1.5   # Max 50% Überschuss durch Schätzung


# ─── SemanticMemory ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_semantic_memory_suche():
    from core.brain_manager import BrainManager, BrainEntry
    from core.brain_index import BrainIndex
    from memory.semantic_memory import SemanticMemory
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        bm = BrainManager(brain_pfad=tmp)
        entry = BrainEntry(id="", typ="fakt", inhalt="Nova ist ein KI-Assistent",
                           quelle="test", vektor=[1.0, 0.0], vertrauen=0.9)
        await bm.add(entry)

        idx = BrainIndex()
        alle = await bm.alle()
        idx.neu_aufbauen(alle)

        sm = SemanticMemory(bm, idx)
        treffer = await sm.suche([0.99, 0.01], max_ergebnisse=3)
        assert len(treffer) >= 1
        assert any("Nova" in t.inhalt for t in treffer)

@pytest.mark.asyncio
async def test_semantic_memory_leerer_vektor():
    from core.brain_manager import BrainManager
    from core.brain_index import BrainIndex
    from memory.semantic_memory import SemanticMemory
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sm = SemanticMemory(BrainManager(tmp), BrainIndex())
        treffer = await sm.suche([])
        assert treffer == []


# ─── Session (Integration) ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_session_baue_kontext():
    from core.brain_manager import BrainManager
    from core.brain_index import BrainIndex
    from core.persona import Persona
    from memory.session import Session
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        pf = Path(tmp) / "persona.yaml"
        pf.write_text("name: Nova\nton: test\nsprache: de\n", encoding="utf-8")
        persona = Persona(pfad=pf)
        bm = BrainManager(brain_pfad=tmp)
        idx = BrainIndex()
        session = Session(persona=persona, brain_manager=bm, brain_index=idx)

        kontext = await session.baue_kontext("Hallo Nova")
        assert len(kontext.messages) >= 1
        assert kontext.messages[-1]["content"] == "Hallo Nova"

@pytest.mark.asyncio
async def test_session_speichere_turn_und_history():
    from core.brain_manager import BrainManager
    from core.brain_index import BrainIndex
    from core.persona import Persona
    from memory.session import Session
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        pf = Path(tmp) / "persona.yaml"
        pf.write_text("name: Nova\nton: test\nsprache: de\n", encoding="utf-8")
        persona = Persona(pfad=pf)
        session = Session(persona=persona, brain_manager=BrainManager(tmp), brain_index=BrainIndex())

        session.speichere_turn("Frage 1", "Antwort 1")
        kontext = await session.baue_kontext("Frage 2")
        # Frage 1 + Antwort 1 + Frage 2
        assert len(kontext.messages) >= 3

def test_session_fact_hinzufuegen():
    from core.brain_manager import BrainManager
    from core.brain_index import BrainIndex
    from core.persona import Persona
    from memory.session import Session
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        pf = Path(tmp) / "persona.yaml"
        pf.write_text("name: Nova\nton: test\nsprache: de\n", encoding="utf-8")
        persona = Persona(pfad=pf)
        session = Session(persona=persona, brain_manager=BrainManager(tmp), brain_index=BrainIndex())

        session.fact_hinzufuegen("User liebt Python", "praeferenz", 0.95)
        assert len(session.session_facts) == 1

@pytest.mark.asyncio
async def test_session_reset():
    from core.brain_manager import BrainManager
    from core.brain_index import BrainIndex
    from core.persona import Persona
    from memory.session import Session
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        pf = Path(tmp) / "persona.yaml"
        pf.write_text("name: Nova\nton: test\nsprache: de\n", encoding="utf-8")
        persona = Persona(pfad=pf)
        session = Session(persona=persona, brain_manager=BrainManager(tmp), brain_index=BrainIndex())

        session.speichere_turn("test", "antwort")
        session.fact_hinzufuegen("fakt")
        session.reset()
        assert len(session.session_facts) == 0
        assert session.episodic.status()["turns"] == 0

