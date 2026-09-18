# Nova Recycling Bundle – 2026-08-24

Arbeitsort: `D:\Nova_Recycling_Bundle_2026-08-24`

Das Bundle sortiert Nova-Komponenten nach sechs Themen: Agentic Systems, Kernel, Local Models, KV-Cache, Layerstreaming und Evolution. Es ist ausdrücklich kein Versuch, eine vollständige Nova-Version zu konservieren.

## Je Themenordner

- `chat_evidence`: ausschließlich gefilterte ZIP-Auszüge
- `code`: zusammenhängende Komponenten und deduplizierte Fragmente
- `inventory`: Treffer, Duplikate und Ursprungspfade
- `tests`: komponentenbezogene Logs und Testquellen
- `notes/EVALUATION.md`: Urteil, Grenzen und Recycling-Empfehlung

Original-Chats existieren nicht. „Implemented“-Behauptungen aus dem ZIP wurden nicht als Buildbeweis gewertet.

## Verifizierte Komponenten

- Nova4 C++: **30/30 CTests bestanden**.
- Gezielte C++-Läufe: Agentic 8/8, Kernel 7/7, Local Models 4/4, KV-Cache 3/3, Layerstreaming 4/4, Evolution 4/4.
- Python Tool-Calling/Skill-Mesh: **21/21 bestanden**.
- Python Layer-1/Memory/Context/Brain: **21/22 bestanden**.
- Python gesamt: **42/43**; der einzelne rote Test widerspricht beim exakten 40-Token-Grenzwert dem dokumentierten `used > target`-Verhalten. Ein Zusatztest oberhalb des Limits bestätigt die Kürzungsfunktion.

Python 3.14.7 ist systemweit installiert. Die Test-venv liegt unter `_global_results\python_component_env`.

## Grenzen

CUDA wird ohne `NOVA_HAVE_CUDA` übersprungen. Mehrere C++-Tests sind synthetisch; RealInferenceKv nutzt ein winziges CPU-Modell. Reales GGUF benötigt `NOVA_REAL_GGUF`. Python-Tooltests mocken Ollama. GPU-Leistung, reale Modellqualität und autonome Evolution sind nicht bewiesen.

| Thema | ZIP-Gruppen | Kernfragmente |
|---|---:|---:|
| Agentic Systems | 440 | 47 |
| Kernel | 264 | 20 |
| Local Models | 434 | 21 |
| KV-Cache | 54 | 8 |
| Layerstreaming | 348 | 10 |
| Evolution | 183 | 34 |

Zuerst die jeweilige Bewertung lesen, danach das Herkunftsmanifest. Zusammenhängende Komponenten bevorzugen; `unique_fragments` ist Ideen-/Vergleichsmaterial.
