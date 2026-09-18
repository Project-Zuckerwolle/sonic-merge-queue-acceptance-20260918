// brain_evergreen.cpp — Implementierung von brain_evergreen.h (Design §11.2 #7).
#include "Brain/brain_evergreen.h"

#include "Brain/wiki_meta.h"

namespace nova::brain {

int mark_evergreen(BrainStore& store, int min_sessions) {
    int marked = 0;
    for (const auto& ref : store.list_wiki()) {
        std::string page;
        if (!store.read_wiki(ref.category, ref.name, page)) continue;
        WikiMeta m = parse_meta(page);
        if (!m.evergreen && int(m.sessions.size()) >= min_sessions) {
            m.evergreen = true;
            m.inactive  = false;  // Evergreen hebt Inaktiv auf
            store.write_wiki(ref.category, ref.name, apply_meta(strip_meta(page), m));
            ++marked;
        }
    }
    return marked;
}

}  // namespace nova::brain
