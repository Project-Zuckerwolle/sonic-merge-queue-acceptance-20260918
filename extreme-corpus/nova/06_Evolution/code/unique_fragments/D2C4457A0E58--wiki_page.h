// wiki_page.h — das Seitenformat des Brain-Wikis und sein Parser.
//
// ERSETZT wiki_meta.h/.cpp. Was dort kaputt war, am Code:
//
//   1. FUSSZEILE STATT KOPF. apply_meta (wiki_meta.cpp:100-121) hängt die
//      Metadaten ans DATEIENDE. Ohne atomares Schreiben (Nova 4 schrieb mit
//      ofstream/trunc, brain_store.cpp:24) geht bei einem Abbruch als erstes
//      das verloren, was am Ende steht — also genau die Metadaten. Wir
//      schreiben jetzt atomar, aber der Kopf bleibt trotzdem vorn: der Indexer
//      braucht Kategorie, Aliase und Links, ohne die ganze Datei zu lesen.
//
//   2. ZWEI PARSER, ZWEI TRENNER, BEIDE ZERBRECHLICH.
//        <!-- Links: ziel:0.9:grund, ziel2:0.7:grund -->
//        parse_meta (wiki_meta.cpp:46-58) trennt Einträge an `,` und Felder an
//        `:`. Eine Begründung wie "Läuft auf: AMD" zerlegt das Link-Feld; ein
//        Komma in der Begründung erzeugt einen Geister-Link.
//        <!-- nova-meta last_mentioned=… sessions=a;b -->
//        wird mit `ss >> tok` an LEERZEICHEN zerlegt (wiki_meta.cpp:66) — eine
//        Session-Kennung mit Leerzeichen (nova_main.cpp:328 baute genau solche:
//        "2026-07-25 03:00" wäre eine) kappt die Liste.
//      LÖSUNG: EIN Trenner `|`, und `|` kommt in Werten nicht vor, weil es beim
//      Schreiben durch `/` ersetzt wird (sanitize_field). Kein Escaping, also
//      auch kein Escape-Parser, der falsch sein kann.
//
//   3. `# [inaktiv]` (wiki_meta.cpp:79, :120) ist eine Markdown-H1 mitten im
//      Dokument. Sie landete im Prompt, sah dort aus wie eine Überschrift und
//      war für jeden Markdown-Renderer ein zweiter Titel. Jetzt: `status|inaktiv`
//      im Kopf.
//
//   4. LINKS OHNE ORT. Die Begründung stand in einem HTML-Kommentar, der
//      Fließtext wusste nichts davon. Jetzt steht der Link da, wo die Begründung
//      steht: `[[ziel]]` im Text. Der Graph ist ein einzeiliger Scan über die
//      Datei; das Feld `reason` und sein Parser entfallen ersatzlos.
//
// FORMAT (neu):
//
//   # Titel der Seite
//   <!--nova
//   category|concepts
//   aliases|Nova 4|Nova-Assistent
//   created|2026-07-01
//   last_mentioned|2026-07-25
//   sessions|s-1|s-2
//   evergreen|1
//   status|aktiv
//   links|inference|0.80|amd-gpu|0.50
//   -->
//
//   Nova läuft seit 2026-07 auf AMD, siehe [[amd-gpu]] (Wechsel von CUDA).  #hardware
//
// Warum der Titel VOR dem Kopfblock steht: memory_tools.cpp:249-252 nimmt die
// erste Zeile einer Wiki-Datei als Titel für `erinnerung_suchen`. Stünde dort
// `<!--nova`, hieße jeder Brain-Treffer im Suchergebnis "<!--nova". Der Kopf
// beginnt in Zeile 2 — für "der Indexer muss nicht die ganze Datei lesen" macht
// eine Zeile keinen Unterschied.
//
// parse() versteht BEIDE Formate. render() schreibt immer das neue. Damit
// migriert der laufende Zyklus den Bestand von selbst — kein Stichtag, kein
// Migrationsskript, keine Version, die irgendwo gepflegt werden muss.
#pragma once

#include "Brain/brain_types.h"

#include <string>
#include <vector>

namespace nova::brain {

// ---------------------------------------------------------------------------
// Kategorien
// ---------------------------------------------------------------------------
enum class Category { Entities, Concepts, Synthesis };

const char* category_dir(Category c);            // "entities" | "concepts" | "synthesis"
const char* category_label(Category c);          // deutsch, für Prompt und Bericht
Category    category_from(const std::string& s); // tolerant; Vorgabe Synthesis

// ---------------------------------------------------------------------------
// Link
// ---------------------------------------------------------------------------
// Kein `reason` mehr: die Begründung steht im Fließtext neben `[[ziel]]`. Das
// Gewicht ist ABGELEITET (Linker) und lebt nur im Kopf, damit der Index es
// ohne Textanalyse hat.
struct PageLink {
    std::string target;
    double      weight = 0.5;
};

// ---------------------------------------------------------------------------
// Seite
// ---------------------------------------------------------------------------
struct WikiPage {
    std::string              id;              // Dateiname ohne ".md" (kebab)
    Category                 category = Category::Synthesis;
    std::string              title;
    std::vector<std::string> aliases;         // löst das Synonymproblem ohne Embeddings
    std::string              created;         // "YYYY-MM-DD"
    std::string              last_mentioned;  // "YYYY-MM-DD"
    std::vector<std::string> sessions;        // distinkte Sitzungs-Kennungen
    bool                     evergreen = false;
    bool                     inactive = false;  // status|inaktiv
    std::vector<PageLink>    links;
    std::vector<std::string> tags;            // `#kennung` aus dem Fließtext
    std::string              body;            // reiner Inhalt, ohne Titelzeile und Kopf

    void add_session(const std::string& id);
    void add_alias(const std::string& a);
    // Erwähnung eintragen. Hebt `inactive` auf — kein Decay, nur die Flagge.
    void touch(const std::string& today, const std::string& session);
    bool has_link(const std::string& target) const;
    // Verschmilzt einen Link statt ihn zu ersetzen (B4: brain_linker.cpp:76
    // setzte `m.links = links` und warf damit jeden früheren Fund weg).
    void merge_link(const std::string& target, double weight);
    // Entfernt Links auf Ziele, die es nicht mehr gibt — die EINZIGE Art, wie
    // ein Link verfällt.
    int  prune_links(const std::vector<std::string>& existing_ids);

    int  degree() const { return int(links.size()); }
};

// ---------------------------------------------------------------------------
// Format
// ---------------------------------------------------------------------------
WikiPage    parse_page(const std::string& id, Category c, const std::string& text);
std::string render_page(const WikiPage& p);

// Erkennt die alte Fassung (HTML-Kommentar-Fußzeile). Öffentlich, weil der
// Compiler die Migration als Ereignis meldet.
bool legacy_format(const std::string& text);

// Einzeiliger Scan über den Graphen: `[[ziel]]` im Fließtext.
std::vector<std::string> scan_links(const std::string& body);
std::vector<std::string> scan_tags(const std::string& body);

// Dateiname aus einem Titel: ASCII-kebab, Umlaute ausgeschrieben. Muss auf
// jedem Dateisystem gültig sein, deshalb streng.
std::string slugify(const std::string& s);

// Fügt dem Fließtext eine Verbindungszeile hinzu, falls noch nicht vorhanden.
// Liefert false, wenn der Link schon im Text stand.
bool append_link_line(std::string& body, const std::string& target, const std::string& reason);

}  // namespace nova::brain
