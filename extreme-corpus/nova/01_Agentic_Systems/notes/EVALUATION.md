# Nova – Agentic Systems

## Zweck und Quellen

Agentik umfasst Tool-Aufrufe, ReAct-Schleifen, Plan-DAG, Subagent-Kontext, Skills und kontrollierte Arbeitsbereiche. Da keine Original-Chats existieren, bilden die 440 thematisch zugeordneten ZIP-Gruppen unter `chat_evidence` die verfügbare Gesprächsgrundlage.

## Gefundene Komponenten

- Nova4: `src/Apex`, `src/Skills` und `apex`.
- Python-v2/v9: Orchestrator-, Core-, Skill- und Testmodule.
- 47 deduplizierte Kernfragmente mit sämtlichen Fundorten im Inventar.

## Testurteil

- Nova4 Agentic: **8/8 bestanden**.
- Python Tool-Calling/Skill-Mesh/Memory-Extraktion: **21/21 bestanden**.
- Gemeinsam mit Evolution-Memory: insgesamt **42/43 Python-Tests bestanden**.
- Der eine rote Test erwartet Kürzung bei genau 40 von 100 Tokens; die dokumentierte Implementierung kürzt nur oberhalb des 40-Token-Ziels. Ein zusätzlicher Lauf mit 220 Tokens entfernte korrekt 17 Einträge und endete bei 33 Tokens. Bewertung: inkonsistente Test-Fixture, kein belegter Codefehler.
- Ollama und das Modell sind in den Tool-Calling-Tests gemockt; bewertet wird die Agenten-/Toolmechanik, nicht Modellqualität.

Details: `tests/PYTHON_COMPONENT_RESULTS.md`.

## Recycling-Empfehlung

Komponentenweise übernehmen: ToolBox/WorkspaceGuard, ReAct-Zustandsmaschine, Plan-DAG, Grammatik, Subagent-Kontext und Python Skill-Mesh sind brauchbare Bausteine. Die Python-Agentik benötigt die separat gebündelte Evolution-Memory-Komponente. Keine LLM-Qualität aus Mock-Tests ableiten.
