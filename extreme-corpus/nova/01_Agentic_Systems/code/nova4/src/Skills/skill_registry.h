// skill_registry.h — Active Skills, Profile, Permission-Check (Design §13.2, §13.10).
//
// Hält alle registrierten Skills (eingebettete Defaults + geladene Nutzer-Skills),
// die Profil-Auswahl (§13.10) und den Permission-Tier-Check (§13.2):
//   Tier 1 READ        -> nie bestätigen
//   Tier 2 WRITE_LOCAL -> konfigurierbar
//   Tier 3 SYSTEM      -> immer bestätigen (nicht abschaltbar)
//   datei_loeschen     -> Tier 2, aber HARTKODIERTE Pflichtbestätigung
#pragma once

#include <map>
#include <string>
#include <vector>

#include "Skills/skill_loader.h"

namespace nova::skills {

struct Profile {
    std::string name;
    std::vector<std::string> tools;   // erlaubte Normal-Tools
    std::vector<std::string> apex;    // erlaubte Apex-Skills
};

class SkillRegistry {
public:
    void add(const Skill& s);
    const Skill* find(const std::string& name) const;
    std::vector<const Skill*> by_type(SkillType t) const;
    size_t size() const { return skills_.size(); }

    // Profile.
    void add_profile(const Profile& p);
    bool set_active_profile(const std::string& name);
    const std::string& active_profile() const { return active_profile_; }
    bool is_active(const std::string& skill_name) const;  // im aktiven Profil?

    // Permission: muss dieser Skill bestätigt werden? tier2_confirm = Config-Flag.
    bool needs_confirmation(const std::string& skill_name, bool tier2_confirm) const;

    // Kompakte System-Prompt-Zeile (§13.5): "Tools: ... | Apex: ...".
    std::string prompt_fragment() const;

    // Lädt die Standard-Skills aus §13.3/§13.4 (eingebettet).
    static SkillRegistry with_defaults();

private:
    std::map<std::string, Skill> skills_;
    std::map<std::string, Profile> profiles_;
    std::string active_profile_;
};

}  // namespace nova::skills
