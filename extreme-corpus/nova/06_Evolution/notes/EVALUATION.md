# Nova – Evolution

## Zweck und Quellen

Dieser Bereich bündelt Gedächtnis, Episoden, Retrieval/BM25 und Brain-Logik. Die ZIP-Evidenz enthält 183 zugeordnete Gruppen.

## Gefundene Komponenten

- Nova4: `src/Brain`, `src/Memory` und Memory-Testdaten.
- Python-v2/v9: getrennte Brain- und Memory-Module.
- 34 deduplizierte Kernfragmente mit sämtlichen Quellpfaden.

## Testurteil

- Nova4 Memory/Retrieval: **4/4 bestanden**.
- Python Layer-1/Memory/Context/Brain: **21/22 bestanden**.
- Einziger roter Python-Test ist eine Grenzwertabweichung der Test-Fixture: bei exakt 40 Tokens und Ziel 40 muss die dokumentierte `while used > target`-Logik nichts entfernen. Mit 220 Tokens entfernt die Komponente korrekt 17 Einträge bis auf 33 Tokens.
- Memory und Agentic wurden als verbundene Komponenten getestet, nicht als vollständige Nova-Anwendung.

Diese Ergebnisse beweisen keine autonome Verbesserung, sichere Selbstmodifikation oder langfristigen Qualitätsgewinn.

## Recycling-Empfehlung

Memory, Episode, Context Builder und BM25 als kleine Bibliotheken recyceln. „Evolution“ erst nach messbaren Offline-Evals verwenden: Baseline, versionierte Änderungen, Rollback, Datenherkunft und Schutz gegen Selbstbestätigung. Status: **Memory/Retrieval funktional; selbständige Evolution UNPROVEN**.
