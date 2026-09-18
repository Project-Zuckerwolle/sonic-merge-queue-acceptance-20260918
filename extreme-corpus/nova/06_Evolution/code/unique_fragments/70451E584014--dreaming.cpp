// dreaming.cpp — Implementierung von dreaming.h (Design §12.4).
#include "Memory/dreaming.h"

namespace nova::memory {

bool run_dreaming(const brain::LlmFn& llm, const std::string& working_memory,
                  IdentityStore& identity, std::string* err) {
    if (!llm) { if (err) *err = "kein LLM-Callback"; return false; }

    // Bisherigen Dreaming-Inhalt als Kontext mitgeben (inkrementelles Update).
    const std::string* prev = identity.find("Dreaming");
    const std::string prompt =
        "Aktualisiere den Dreaming-Block aus der Session. Extrahiere dauerhafte "
        "Fakten/Präferenzen/Projektstände. Gib NUR den neuen Block-Inhalt zurück.\n\n"
        "--- BISHER ---\n" + (prev ? *prev : std::string()) +
        "\n\n--- SESSION ---\n" + working_memory;

    const std::string new_block = llm(prompt);

    // write_dreaming schreibt ausschließlich in # Dreaming, nie in # LOCKED.
    if (!identity.write_dreaming(new_block)) { if (err) *err = "write_dreaming verweigert"; return false; }
    return true;
}

}  // namespace nova::memory
