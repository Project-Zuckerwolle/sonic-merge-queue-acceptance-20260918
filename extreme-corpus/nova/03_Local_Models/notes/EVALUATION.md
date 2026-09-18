# Nova – Local Models

## Zweck und Ideenbild

Die ZIP-Analyse beschreibt lokale Modellablage, Tokenizer/Modellformate, Konvertierung und First-Token-Inferenz. 434 passende Gesprächsgruppen wurden aus dem ZIP selektiert; Original-Chats existieren laut Nutzer nicht.

## Gefundene Umsetzung

- Nova4 `src/ModelStore` und `src/InferEngine`.
- Quasar Modell- und Tokenizer-Implementierungen samt Headern.
- 21 inhaltlich eindeutige Kernfragmente mit Herkunftsmanifest.

## Testurteil

**Funktional im begrenzten Umfang:** 4/4 ausgewählte CTests bestanden: ChunkRead, ConverterSelftest, FirstToken und RealInferenceKv.

**Substanz:** RealInferenceKv baut ein deterministisches winziges synthetisches Modell und prüft CPU-Inferenz plus KV-Verhalten. ConverterSelftest und ChunkRead prüfen konkrete Datenpfade.

**Grenzen:** FirstToken ist ein Mock-Test. Ein reales GGUF/24B-Modell wird nur getestet, wenn `NOVA_REAL_GGUF` gesetzt ist; das war hier nicht der Fall. GPU-Abschnitte sind ebenfalls nicht aktiv. Somit sind reale Modellkompatibilität, Antwortqualität, Speicherbedarf und Durchsatz **UNTESTED**.

## Recycling-Empfehlung

ModelStore, Chunk-I/O, Konverter-Selbsttest und deterministischen Mini-Modelltest gemeinsam sichern. Quasar-Tokenizer/Modellcode als Vergleichsimplementation behalten. Nächster Gate-Test muss ein frei wählbares kleines echtes GGUF umfassen, danach erst große Modelle und GPU.
