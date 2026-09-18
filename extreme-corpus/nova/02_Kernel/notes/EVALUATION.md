# Nova – Kernel

## Zweck und Ideenbild

Das Dokument bündelt hier Rechenkerne für Dekompression, Quantisierung/GEMM, speculative decoding und Medusa-artige Verifikation. Die ZIP-basierte Gesprächsevidenz enthält 264 zugeordnete Gruppen und ist unter `chat_evidence` unverändert abgelegt.

## Gefundene Umsetzung

- Nova4 `src/InferEngine`: zusammenhängender Inferenzpfad mit Kernel-nahen Komponenten.
- Quasar `src/kernels`: zusätzliche spezialisierte Kernelfragmente.
- 20 inhaltlich eindeutige Kernfragmente wurden dedupliziert und mit sämtlichen Ursprungspfaden manifestiert.

## Testurteil

**Funktional auf CPU/synthetisch:** 7/7 ausgewählte Tests bestanden: Decompress, TurboQuant, SpecDecode, Medusa, CUDA-Testhülle, FusedInt3Gemm und MedusaTreeVerify.

**Wichtige Grenzen:** Der CUDA-Test beendet sich ohne `NOVA_HAVE_CUDA` als „SKIPPED“ mit Exitcode 0; damit ist keine echte GPU-Ausführung belegt. TurboQuant und mehrere Kerneltests verwenden kleine synthetische Daten. Der grüne Lauf beweist Kontrollfluss, Datenformate und CPU-/Fallback-Verhalten, nicht Performance oder numerische Qualität auf Ziel-GPU und großem Modell.

## Recycling-Empfehlung

Nova4 InferEngine als Integrationsbasis verwenden; Quasar-Kernel nur einzeln übernehmen, wenn ABI, Layout und Lizenz/Provenienz geklärt sind. Vor Produktion zwingend GPU-CI ergänzen: echte CUDA/HIP-Kompilierung, Referenzvergleich, Grenzfälle und Benchmarks. Aktueller Status: **funktionaler Prototyp, GPU-Leistung UNTESTED**.
