# Nova – KV-Cache

## Zweck und Ideenbild

Der Bericht behandelt KV-Persistenz, Prefix-Wiederverwendung und das Zusammenspiel mit Quantisierung/Inferenz. Die ZIP-Evidenz umfasst 54 eng zugeordnete Gruppen.

## Gefundene Umsetzung

- Nova4 InferEngine mit KV-bezogenen Pfaden.
- Quasar `src/kv`.
- 8 inhaltlich eindeutige Kernfragmente; Duplikate und alle Fundorte sind im Manifest dokumentiert.

## Testurteil

**Funktional auf synthetischer CPU-Basis:** 3/3 Tests bestanden: TurboQuant, KvPrefixCache und RealInferenceKv. Prefix-Treffer und deterministische Inferenzzustände werden tatsächlich ausgeführt.

**Grenzen:** Kein großer realer Modelllauf, keine GPU-Ausführung, keine Langzeit-/Mehrbenutzerlast und keine belastbare Messung der Speichereinsparung. TurboQuant ist ein synthetischer Surrogat-Test.

## Recycling-Empfehlung

Prefix-Keying, Cache-Lifecycle und der deterministische RealInferenceKv-Test als Einheit übernehmen. Vor Zusammenführung Invalidation, Modell-/Tokenizer-Fingerprints, Kontextgrenzen, Parallelität und persistente Cache-Korruption testen. Status: **funktionale Kernidee, Produktionsskalierung UNTESTED**.
