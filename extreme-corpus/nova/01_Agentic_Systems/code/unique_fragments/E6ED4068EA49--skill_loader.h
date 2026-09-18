// skill_loader.h — Unified-YAML-Skill-Format (Design §13.1).
//
// ALLE Skills (Normal-Tools type:tool + Apex type:apex) nutzen dasselbe YAML.
// Das type-Feld unterscheidet sie. YAMLs sind in nova4.exe eingebettet
// (programmatisch via SkillRegistry) oder in %AppData%\Nova4\skills\ (Nutzer).
#pragma once

#include <string>
#include <vector>

namespace nova::skills {

enum class SkillType { Tool, Apex };

// Permission-Tier-Modell (§13.2).
enum Tier { TIER_READ = 1, TIER_WRITE_LOCAL = 2, TIER_SYSTEM = 3 };

struct SkillParam {
    std::string name;
    std::string type;       // string/int/bool/...
    bool        required = false;
};

struct Skill {
    std::string             name;
    SkillType               type = SkillType::Tool;
    int                     tier = TIER_READ;
    std::string             description;
    std::vector<std::string> trigger_keywords;
    std::string             executor;            // native / 14B / 3B
    std::vector<SkillParam>  parameters;

    // Nur type: apex
    bool        react_loop = false;
    int         max_iterations = 20;
    bool        persistent = false;
    std::string template_path;
    std::string subagent_executor;
};

// Parst eine einzelne Skill-YAML (Subset: key: value, [inline lists], - items).
bool parse_skill_yaml(const std::string& yaml, Skill& out, std::string* err = nullptr);

// Lädt alle *.yaml/*.yml aus einem Verzeichnis.
std::vector<Skill> load_skill_dir(const std::string& dir);

}  // namespace nova::skills
