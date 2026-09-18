// brain_evergreen.h — Evergreen-Flag (Design §11.2 #7).
//
// Seiten die in >= 3 verschiedenen Sessions erwähnt wurden bekommen
// <!-- evergreen -->. Evergreen-Seiten werden bei der Konfidenz-Konsolidierung
// nie als [inaktiv] markiert, unabhängig von der letzten Erwähnung.
// Läuft täglich (03:00) als Teil des Brain-Zyklus.
#pragma once

#include "Brain/brain_store.h"

namespace nova::brain {

// Setzt das Evergreen-Flag für Seiten mit >= min_sessions distinkten Sessions.
// Liefert die Anzahl neu als evergreen markierter Seiten.
int mark_evergreen(BrainStore& store, int min_sessions = 3);

}  // namespace nova::brain
