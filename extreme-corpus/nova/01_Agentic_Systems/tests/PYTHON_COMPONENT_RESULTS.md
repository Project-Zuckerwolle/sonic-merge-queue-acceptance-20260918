# Python-Komponententests

## Umgebung

- Systeminterpreter: Python 3.14.7, 64 Bit
- Pfad: `C:\Program Files\Python314\python.exe`
- Isolierte venv: `D:\Nova_Recycling_Bundle_2026-08-24\_global_results\python_component_env`
- Ergänzte Testabhängigkeiten: pytest, pytest-asyncio, numpy, PyYAML, aiofiles

## Komponentenverknüpfung

Die Tests in `01_Agentic_Systems\code\python_nova_v2` benötigen sowohl Agentic-Module als auch die Memory-Komponente aus `06_Evolution\code\python_nova_v2`. Getestet wurde mit beiden Verzeichnissen im `PYTHONPATH`; die Komponenten wurden nicht zu einer vollständigen Nova-Anwendung zusammengebaut.

## Ergebnisse

- Tool-Calling/Skill-Mesh/Memory-Extraktion: **21/21 bestanden**. Ollama-Aufrufe sind in diesen Tests ausdrücklich gemockt.
- Layer-1/Memory/Context/Brain: **21/22 bestanden**.
- Gesamt: **42/43 bestanden**.
- Logs: `python-components-pytest.log`, `python-layer1-pytest.log`.

## Verbleibende Abweichung

`test_episodic_kuerzen` erwartet, dass bei exakt 40 geschätzten Tokens und Ziel `40 % von 100 = 40` Einträge entfernt werden. Die Implementierung dokumentiert und verwendet dagegen die Bedingung „entfernen, solange genutzt > Ziel“. Bei Gleichheit entfernt sie korrekt 0.

Der zusätzliche Diagnoseversuch bestätigt die Funktion:

- kurze Daten: 40 Tokens, entfernt 0;
- lange Daten: 220 Tokens, entfernt 17;
- danach 33 Tokens und damit unter dem 40-Token-Ziel.

Bewertung: **Test-Fixture/Erwartung inkonsistent mit dem dokumentierten Grenzverhalten**, nicht ein nachgewiesener Implementierungsdefekt. Der Originalcode wurde nicht verändert.
