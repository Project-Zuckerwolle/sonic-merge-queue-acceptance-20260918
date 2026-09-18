# Nova – Layerstreaming

## Zweck und Ideenbild

Die ZIP-Auswertung beschreibt schichtweises Laden, Triple Buffering, Prefetching und einen VRAM-Floor zur Inferenz mit begrenztem Speicher. 348 zugeordnete Gesprächsgruppen liegen als Evidenz vor.

## Gefundene Umsetzung

- Nova4 `src/InferEngine` und `src/System`.
- 10 inhaltlich eindeutige Kernfragmente mit vollständigem Herkunftsmanifest.
- Zusammenhängende Testquellen für Chunk-I/O, Triple Buffer, Prefetch und VRAM-Floor.

## Testurteil

**Funktional für Ablauf/Fallback:** 4/4 Tests bestanden: ChunkRead, TripleBuffer, PrefetchVram und VramFloor.

**Grenzen:** Der Serverlauf belegt keine reale CUDA-/HIP-Schichtübertragung und keinen echten VRAM-Druck. Ohne Ziel-GPU und echtes Modell bleiben Überlappung von I/O/Compute, Bandbreite, Pinning, OOM-Verhalten und erreichbare Tokenrate **UNTESTED**.

## Recycling-Empfehlung

Scheduler-Zustände, Triple-Buffer-Protokoll, Chunk-Leser und VRAM-Floor zusammen recyceln; nicht nur einzelne Kernel kopieren. Danach Hardwaretest mit Telemetrie für Transferzeiten, Stall-Anteil, Peak-VRAM und korrekte Layer-Reihenfolge.
