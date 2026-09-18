// skill_registry.cpp — Implementierung von skill_registry.h (Design §13.2, §13.10).
#include "Skills/skill_registry.h"

#include <algorithm>

namespace nova::skills {

void SkillRegistry::add(const Skill& s) { skills_[s.name] = s; }

const Skill* SkillRegistry::find(const std::string& name) const {
    auto it = skills_.find(name);
    return it == skills_.end() ? nullptr : &it->second;
}

std::vector<const Skill*> SkillRegistry::by_type(SkillType t) const {
    std::vector<const Skill*> out;
    for (const auto& [n, s] : skills_) if (s.type == t) out.push_back(&s);
    return out;
}

void SkillRegistry::add_profile(const Profile& p) { profiles_[p.name] = p; }

bool SkillRegistry::set_active_profile(const std::string& name) {
    if (!profiles_.count(name)) return false;
    active_profile_ = name;
    return true;
}

bool SkillRegistry::is_active(const std::string& skill_name) const {
    if (active_profile_.empty()) return true;  // kein Profil -> alles aktiv
    auto it = profiles_.find(active_profile_);
    if (it == profiles_.end()) return true;
    const Profile& p = it->second;
    return std::find(p.tools.begin(), p.tools.end(), skill_name) != p.tools.end() ||
           std::find(p.apex.begin(), p.apex.end(), skill_name) != p.apex.end();
}

bool SkillRegistry::needs_confirmation(const std::string& skill_name, bool tier2_confirm) const {
    // datei_loeschen: harte Pflichtbestätigung unabhängig von Tier-2-Config (§13.2).
    if (skill_name == "datei_loeschen") return true;
    const Skill* s = find(skill_name);
    if (!s) return true;  // Unbekannt -> sicherheitshalber bestätigen
    switch (s->tier) {
        case TIER_READ:        return false;
        case TIER_WRITE_LOCAL: return tier2_confirm;
        case TIER_SYSTEM:      return true;          // nicht abschaltbar
        default:               return true;
    }
}

std::string SkillRegistry::prompt_fragment() const {
    std::string tools = "Tools: ", apex = "Apex: ";
    bool ft = true, fa = true;
    for (const auto& [n, s] : skills_) {
        if (!is_active(n)) continue;
        if (s.type == SkillType::Tool) { if (!ft) tools += ", "; tools += n; ft = false; }
        else { if (!fa) apex += ", "; apex += n; fa = false; }
    }
    return tools + "\n" + apex + "\nNutze <tool_call> oder <apex_call> wenn sinnvoll.";
}

namespace {
Skill tool(const std::string& name, int tier, const std::string& desc,
           std::vector<std::string> kw) {
    Skill s; s.name = name; s.type = SkillType::Tool; s.tier = tier;
    s.description = desc; s.trigger_keywords = std::move(kw); s.executor = "native";
    return s;
}
Skill apex(const std::string& name, const std::string& desc, std::vector<std::string> kw,
           const std::string& tmpl) {
    Skill s; s.name = name; s.type = SkillType::Apex; s.tier = TIER_WRITE_LOCAL;
    s.description = desc; s.trigger_keywords = std::move(kw); s.executor = "14B";
    s.react_loop = true; s.max_iterations = 20; s.persistent = true;
    s.template_path = tmpl; s.subagent_executor = "3B";
    return s;
}
}  // namespace

SkillRegistry SkillRegistry::with_defaults() {
    SkillRegistry r;
    // Normal-Tools (§13.3).
    r.add(tool("wetter", TIER_READ, "Open-Meteo, kein API-Key", {"wetter"}));
    r.add(tool("websearch", TIER_READ, "DuckDuckGo/SearXNG", {"suche"}));
    r.add(tool("news_rss", TIER_READ, "RSS-Feeds", {"news"}));
    r.add(tool("finanzen", TIER_READ, "Kurs-APIs", {"kurs"}));
    r.add(tool("rechner", TIER_READ, "Mathematische Ausdrücke", {"rechne"}));
    r.add(tool("uhrzeit", TIER_READ, "Aktuelle Zeit/Datum", {"uhrzeit"}));
    r.add(tool("codebase_scan", TIER_READ, "File-Tree bis 200 Dateien", {"scan"}));
    r.add(tool("todo_lesen", TIER_READ, "Todo-Liste lesen", {"todos"}));
    r.add(tool("datei_lesen", TIER_READ, "Datei/Verzeichnis laden", {"lies"}));
    r.add(tool("datei_suchen", TIER_READ, "Suche nach Name/Typ/Datum", {"finde datei"}));
    r.add(tool("todo_schreiben", TIER_WRITE_LOCAL, "Todos erstellen/abhaken", {"todo"}));
    r.add(tool("datei_schreiben", TIER_WRITE_LOCAL, "Datei erstellen/überschreiben", {"schreibe"}));
    r.add(tool("datei_verschieben", TIER_WRITE_LOCAL, "Verschieben/umbenennen", {"verschiebe"}));
    r.add(tool("datei_sortieren", TIER_WRITE_LOCAL, "Nach Kriterien sortieren", {"sortiere"}));
    r.add(tool("ordner_erstellen", TIER_WRITE_LOCAL, "Verzeichnis erstellen", {"ordner"}));
    r.add(tool("datei_loeschen", TIER_WRITE_LOCAL, "Löschen — Pflichtbestätigung", {"lösche"}));
    r.add(tool("git_ops", TIER_WRITE_LOCAL, "status/diff/log/add/commit", {"git"}));
    r.add(tool("shell_exec", TIER_SYSTEM, "CreateProcess, Workspace-Sandbox, 60s", {"shell"}));
    r.add(tool("power_sleep", TIER_SYSTEM, "SetSuspendState(S4)", {"sleep"}));

    // Apex-Skills (§13.4).
    r.add(apex("coding", "Coding-Projekte: planen, schreiben, builden, testen",
               {"code schreiben", "implementiere", "fix", "refactor"}, "apex/coding_prompt.md"));
    r.add(apex("news_research", "News recherchieren", {"news", "recherchiere"},
               "apex/news_research_prompt.md"));
    r.add(apex("deep_research", "Tiefe Recherche", {"recherchiere ausführlich"},
               "apex/deep_research_prompt.md"));
    r.add(apex("datei_projekt", "Dateien organisieren", {"projekt aufräumen"}, ""));

    // Profile (§13.10).
    r.add_profile({"profile_coding",
        {"datei_lesen", "datei_schreiben", "datei_suchen", "codebase_scan", "websearch",
         "git_ops", "shell_exec", "todo_lesen", "todo_schreiben"},
        {"coding", "datei_projekt"}});
    r.add_profile({"profile_research",
        {"websearch", "news_rss", "finanzen", "datei_schreiben"},
        {"news_research", "deep_research"}});
    r.add_profile({"profile_assistant", {"wetter", "todo_lesen", "todo_schreiben", "uhrzeit", "datei_schreiben"}, {}});
    r.add_profile({"profile_minimal", {}, {}});
    return r;
}

}  // namespace nova::skills
