// dreaming.h — Dreaming nach jeder Session (Design §12.4, §10.8).
//
// Ministral 14B liest das Working Memory und schreibt in den # Dreaming-Block
// von identity.md. # LOCKED-Blöcke werden NIEMALS angefasst (Garantie über
// IdentityStore::write_dreaming). Session-Facts werden hier extrahiert.
#pragma once

#include <string>

#include "Brain/brain_types.h"
#include "Memory/identity_store.h"

namespace nova::memory {

// Aktualisiert den Dreaming-Block aus dem Working Memory. Liefert false wenn der
// LLM-Callback fehlt oder das Schreiben scheitert (LOCKED bleibt unberührt).
bool run_dreaming(const brain::LlmFn& llm, const std::string& working_memory,
                  IdentityStore& identity, std::string* err = nullptr);

}  // namespace nova::memory
