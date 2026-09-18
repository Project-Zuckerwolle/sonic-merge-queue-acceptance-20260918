import fs from "node:fs";
import path from "node:path";

const root = "C:/Users/user/Documents/Codex/2026-08-23/da/work/claude-extracted-2026-08-24";
const output = "C:/Users/user/Documents/Codex/2026-08-23/da/work/nova_apex_mechanical_analysis.json";

const stop = new Set(`aber alle allem allen aller alles also als am an and auch auf aus bei bin bis bist da damit dann das dass dein deine dem den der des die dies diese doch dort du durch ein eine einem einen einer eines er es etwas für fuer gegen gewesen hat hatte haben hier ich im in ist ja jede jedem jeden jeder jedes kann kein keine mit muss nach nicht nichts noch nun nur ob oder of on ohne sein seine selbst sich sie sind so the to über um und uns unter vom von vor war waren was weil weiter welche wenn werden wie wieder will wir wird wo zu zum zur sowie zwar bitte jetzt schon sehr mehr genau machen gemacht macht sollte sollen soll könnte könnten mochte möchte möchten wollen wollte projekt projekte system funktion funktionen idee ideen plan chat claude user assistant antwort frage file datei code tool tools`.split(/\s+/));

const taxonomy = {
  agentic_systems: ["agentic", "agent", "agents", "subagent", "sub-agent", "orchestrator", "orchestration", "react loop", "react-loop", "planner", "planning", "router", "skillregistry", "skill registry", "tool selection", "tool-selection", "apex", "claw", "autonom", "autonomous"],
  kernel: ["kernel", "hip kernel", "rocm", "cuda", "triton", "custom kernel", "flash attention", "attention kernel", "gpu kernel", "compiled", "compiler"],
  local_models: ["local model", "lokales modell", "lokale modelle", "ollama", "llama.cpp", "gguf", "quant", "quantisierung", "gemma", "ministral", "mistral", "qwen", "3b", "7b", "14b", "30b", "model routing", "modell routing"],
  kv_cache: ["kv cache", "kv-cache", "kvcache", "key value cache", "paged attention", "prefix cache", "context cache", "cache quant", "cache offload", "context window", "kontextfenster"],
  layerstreaming: ["layerstream", "layer streaming", "layer-streaming", "layered streaming", "gelayerstreamed", "layer stream", "weight streaming", "layer offload", "gpu offload", "cpu offload", "vram", "swap", "layers laden", "schichten laden"],
  evolution: ["evolution", "self improvement", "self-improvement", "selbstverbesser", "finetune", "fine-tune", "finetuning", "training", "nightly", "feedback loop", "learning loop", "skill evolution", "memory", "episodic", "semantic memory", "workingmemory", "drift", "eval", "benchmark"],
};

const categoryLabels = {
  agentic_systems: "Agentic Systems",
  kernel: "Kernel",
  local_models: "Local Models",
  kv_cache: "KV-Cache",
  layerstreaming: "Layerstreaming",
  evolution: "Evolution",
};

const contextRegex = /\b(nova|apex|claw|orchestrator|sub-?agent|layerstream|kv[- ]?cache|kernel|local model|lokale[sn]? modell|ollama|gemma|ministral|mistral|vram|rocm|cuda|evolution)\b/i;
const planRegex = /\b(soll|sollte|möchte|moechte|muss|geplant|planen|bauen|implementieren|integrieren|überarbeiten|umbauen|entwickeln|designen|brauchen|fehlt)\b/i;
const implementedRegex = /\b(implementiert|integriert|fertig|erledigt|funktioniert|vorhanden|existiert|erstellt|gefixt|behoben|produktiv|läuft|laeuft|working|done)\b/i;
const rejectedRegex = /\b(verworfen|gestrichen|entfernt|scrap|scrappe|abgelehnt|nicht mehr|doch nicht|deprecated|rausnehmen)\b/i;
const claimRegex = /\b(kann|unterstützt|unterstuetzt|ist stabil|production|produktionsreif|vollständig|vollstaendig|perfekt|revolutionär|revolutionaer|polished|robust|skalierbar)\b/i;

const evidencePatterns = {
  code_path: /(?:[A-Za-z]:\\|\/)[\w .\-\/\\]+\.(?:py|js|jsx|ts|tsx|cpp|cu|hip|json|md)|\b[\w.-]+\.(?:py|js|jsx|ts|tsx|cpp|cu|hip)\b/i,
  test: /\b(test|tests|pytest|benchmark|eval|evaluation|suite|assert|passed|failed)\b/i,
  metric: /\b\d+(?:[.,]\d+)?\s*(?:ms|s|sek|tokens?\/s|tok\/s|gb|mb|%|x|fps|tflops|watt|w)\b/i,
  log: /\b(log|traceback|exception|error code|fehlercode|\[ok\]|\[error\]|info\s+nova)\b/i,
  version: /\b(v\d+(?:\.\d+)*|version\s*\d+|commit|sha|branch)\b/i,
};

const riskPatterns = {
  unbounded_autonomy: /\b(stundenlang|autonom|ohne überwachung|ohne ueberwachung|selbstständig|selbststaendig|immer läuft|immer laeuft|unbeaufsichtigt)\b/i,
  self_modification: /\b(sich selbst|eigenen code|selbst.*ändern|selbst.*aendern|self[- ]modif|evolution|finetun|weiter train)\b/i,
  destructive_actions: /\b(löschen|loeschen|entfernen|aufräumen|aufraeumen|delete|overwrite|überschreiben|ueberschreiben)\b/i,
  security_boundary: /\b(tunnel|remote|fernzugriff|zugriff|computer control|computer-control|bash|shell|admin|permission|sicher)\b/i,
  resource_pressure: /\b(vram|ram|swap|offload|thermal|hitze|power|strom|gpu|cpu|speicher)\b/i,
  training_drift: /\b(drift|continual|kontinuierlich|jede nacht|nightly|online training|fortlaufend.*train)\b/i,
};

function normalize(text) {
  return String(text || "").toLowerCase().normalize("NFKD").replace(/[\u0300-\u036f]/g, "").replace(/[^a-z0-9+#.-]+/g, " ").trim();
}

function terms(text) {
  return normalize(text).split(/\s+/).filter((t) => t.length >= 3 && t.length <= 35 && !stop.has(t) && !/^\d+$/.test(t));
}

function collectStrings(value, key = "", depth = 0, out = []) {
  if (value == null || depth > 10) return out;
  if (typeof value === "string") {
    if (/uuid|signature|approval|icon|timestamp|created_at|updated_at|url/i.test(key)) return out;
    if (value.length >= 3 && !/^https?:\/\//i.test(value) && !/^[A-Za-z0-9+/=]{500,}$/.test(value)) out.push(value);
    return out;
  }
  if (Array.isArray(value)) {
    for (const item of value) collectStrings(item, key, depth + 1, out);
  } else if (typeof value === "object") {
    for (const [childKey, childValue] of Object.entries(value)) {
      if (/^(uuid|id|parent_message_uuid|flags|approval_key|approval_key_legacy|integration_icon_url)$/i.test(childKey)) continue;
      collectStrings(childValue, childKey, depth + 1, out);
    }
  }
  return out;
}

function splitParagraphs(text) {
  return String(text || "").split(/\n{2,}|(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9])/).map((s) => s.replace(/\s+/g, " ").trim()).filter((s) => s.length >= 35 && s.length <= 1800);
}

function categoriesFor(text) {
  const n = normalize(text);
  return Object.entries(taxonomy).filter(([, keywords]) => keywords.some((keyword) => n.includes(normalize(keyword)))).map(([key]) => key);
}

function evidenceFor(text) {
  return Object.entries(evidencePatterns).filter(([, regex]) => regex.test(text)).map(([key]) => key);
}

function risksFor(text) {
  return Object.entries(riskPatterns).filter(([, regex]) => regex.test(text)).map(([key]) => key);
}

function statusFor(text) {
  if (rejectedRegex.test(text)) return "rejected_or_reversed";
  if (implementedRegex.test(text)) return "implemented_claim";
  if (planRegex.test(text)) return "planned";
  if (claimRegex.test(text)) return "capability_claim";
  return "context_or_discussion";
}

const units = [];
const conversations = JSON.parse(fs.readFileSync(path.join(root, "conversations", "conversations.json"), "utf8"))
  .sort((a, b) => String(a.created_at).localeCompare(String(b.created_at)));

for (const [conversationIndex, conversation] of conversations.entries()) {
  const messages = [...(conversation.chat_messages || [])].sort((a, b) => String(a.created_at).localeCompare(String(b.created_at)));
  for (const [messageIndex, message] of messages.entries()) {
    const raw = collectStrings(message).join("\n");
    for (const paragraph of splitParagraphs(raw)) {
      if (!contextRegex.test(paragraph)) continue;
      const categories = categoriesFor(paragraph);
      if (!categories.length) continue;
      units.push({
        sourceType: "conversation",
        sourceId: conversation.uuid,
        sourceOrder: conversationIndex + 1,
        sourceTitle: conversation.name || "Ohne Titel",
        sourceFile: "conversations.json",
        date: message.created_at || conversation.created_at || "",
        sender: message.sender || "",
        position: messageIndex + 1,
        text: paragraph,
        categories,
      });
    }
  }
}

const projectDir = path.join(root, "projects", "projects");
for (const filename of fs.readdirSync(projectDir).filter((name) => name.endsWith(".json"))) {
  const project = JSON.parse(fs.readFileSync(path.join(projectDir, filename), "utf8"));
  for (const [docIndex, doc] of (project.docs || []).entries()) {
    const raw = `${doc.filename || ""}\n${doc.content || ""}`;
    for (const paragraph of splitParagraphs(raw)) {
      if (!contextRegex.test(paragraph)) continue;
      const categories = categoriesFor(paragraph);
      if (!categories.length) continue;
      units.push({
        sourceType: "project_doc",
        sourceId: project.uuid,
        sourceOrder: docIndex + 1,
        sourceTitle: project.name || "Ohne Projektnamen",
        sourceFile: doc.filename || filename,
        date: doc.created_at || project.created_at || "",
        sender: "document",
        position: docIndex + 1,
        text: paragraph,
        categories,
      });
    }
  }
}

const enriched = units.map((unit, index) => {
  const evidence = evidenceFor(unit.text);
  const risks = risksFor(unit.text);
  const status = statusFor(unit.text);
  const tokenSet = new Set(terms(unit.text));
  const evidenceScore = Math.min(100, evidence.length * 18 + (status === "implemented_claim" ? 8 : 0) + (unit.sourceType === "project_doc" ? 7 : 0));
  return { id: index + 1, ...unit, status, evidence, risks, evidenceScore, tokenSet };
});

function jaccard(a, b) {
  let common = 0;
  for (const token of a) if (b.has(token)) common++;
  return common / (a.size + b.size - common || 1);
}

const groups = [];
for (const item of enriched.sort((a, b) => b.evidenceScore - a.evidenceScore || String(b.date).localeCompare(String(a.date)))) {
  let best = null;
  let bestSim = 0;
  for (const group of groups) {
    if (!item.categories.some((c) => group.categories.has(c))) continue;
    const sim = jaccard(item.tokenSet, group.tokenSet);
    if (sim > bestSim) { bestSim = sim; best = group; }
  }
  if (best && bestSim >= 0.56) {
    best.items.push(item);
    for (const category of item.categories) best.categories.add(category);
  } else {
    groups.push({ representative: item, tokenSet: item.tokenSet, categories: new Set(item.categories), items: [item] });
  }
}

const featureGroups = groups.map((group, index) => {
  const statuses = [...new Set(group.items.map((item) => item.status))];
  const evidences = [...new Set(group.items.flatMap((item) => item.evidence))];
  const risks = [...new Set(group.items.flatMap((item) => item.risks))];
  const sources = [...new Set(group.items.map((item) => `${item.sourceType}:${item.sourceOrder}:${item.sourceFile}`))];
  const dates = group.items.map((item) => item.date).filter(Boolean).sort();
  const contradiction = statuses.includes("implemented_claim") && (statuses.includes("planned") || statuses.includes("rejected_or_reversed"));
  const claimWithoutEvidence = statuses.some((s) => s === "implemented_claim" || s === "capability_claim") && evidences.length === 0;
  const recurrence = group.items.length;
  const confidence = Math.max(0, Math.min(100, Math.round(
    20 + Math.min(25, recurrence * 4) + Math.min(30, evidences.length * 8) + Math.min(15, sources.length * 3)
    - (claimWithoutEvidence ? 20 : 0) - (contradiction ? 12 : 0)
  )));
  return {
    groupId: index + 1,
    categories: [...group.categories],
    categoryLabels: [...group.categories].map((c) => categoryLabels[c]),
    representative: group.representative.text.slice(0, 1200),
    recurrence,
    sourceCount: sources.length,
    sources,
    firstSeen: dates[0] || "",
    lastSeen: dates.at(-1) || "",
    statuses,
    evidences,
    risks,
    contradiction,
    claimWithoutEvidence,
    confidence,
  };
}).sort((a, b) => b.confidence - a.confidence || b.recurrence - a.recurrence);

const categorySummary = {};
for (const [category, label] of Object.entries(categoryLabels)) {
  const categoryUnits = enriched.filter((item) => item.categories.includes(category));
  const categoryGroups = featureGroups.filter((group) => group.categories.includes(category));
  const sourceIds = new Set(categoryUnits.map((item) => `${item.sourceType}:${item.sourceId}`));
  categorySummary[category] = {
    label,
    rawMatches: categoryUnits.length,
    featureGroups: categoryGroups.length,
    sources: sourceIds.size,
    implementedClaims: categoryUnits.filter((item) => item.status === "implemented_claim").length,
    planned: categoryUnits.filter((item) => item.status === "planned").length,
    rejectedOrReversed: categoryUnits.filter((item) => item.status === "rejected_or_reversed").length,
    claimsWithoutEvidence: categoryGroups.filter((group) => group.claimWithoutEvidence).length,
    contradictions: categoryGroups.filter((group) => group.contradiction).length,
    riskMentions: categoryUnits.filter((item) => item.risks.length).length,
  };
}

const statusTimeline = enriched
  .filter((item) => item.status !== "context_or_discussion")
  .sort((a, b) => String(a.date).localeCompare(String(b.date)))
  .map((item) => ({
    date: item.date,
    sourceType: item.sourceType,
    sourceOrder: item.sourceOrder,
    sourceTitle: item.sourceTitle,
    sourceFile: item.sourceFile,
    categories: item.categories,
    status: item.status,
    evidence: item.evidence,
    risks: item.risks,
    text: item.text.slice(0, 900),
  }));

const result = {
  generatedAt: new Date().toISOString(),
  method: {
    modelUsedForExtraction: false,
    search: "Keyword taxonomy + recursive export-field extraction",
    grouping: "Jaccard deduplication within overlapping categories",
    confidence: "Mechanical recurrence/evidence score with contradiction and unsupported-claim penalties",
    caveat: "A high score means repeated and evidenced in the export, not that the architecture is technically sound.",
  },
  totals: {
    conversations: conversations.length,
    projectFiles: fs.readdirSync(projectDir).filter((name) => name.endsWith(".json")).length,
    rawMatches: enriched.length,
    featureGroups: featureGroups.length,
    implementedClaims: enriched.filter((item) => item.status === "implemented_claim").length,
    planned: enriched.filter((item) => item.status === "planned").length,
    rejectedOrReversed: enriched.filter((item) => item.status === "rejected_or_reversed").length,
    claimsWithoutEvidence: featureGroups.filter((group) => group.claimWithoutEvidence).length,
    contradictions: featureGroups.filter((group) => group.contradiction).length,
  },
  categorySummary,
  featureGroups,
  statusTimeline,
};

fs.writeFileSync(output, JSON.stringify(result, null, 2), "utf8");
console.log(JSON.stringify({ totals: result.totals, categories: result.category