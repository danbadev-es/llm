// Small formatting helpers shared by the components.

// The line under an answer, from the stats the API sends (live) or stores (history).
export function formatStats(stats) {
  if (!stats) return "";
  if (stats.stopped === "user") return "Detenido";
  const parts = [`${stats.tokens} tokens`, `${Math.round(stats.tok_per_s)} tok/s`, `contexto ${stats.prompt_tokens}`];
  if (stats.stopped === "max_tokens") parts.push("límite de tokens");
  if (stats.dropped_turns) parts.push(`${stats.dropped_turns} turnos antiguos omitidos`);
  return parts.join(" · ");
}

// History groups like ChatGPT/Claude: Hoy, Ayer, Últimos 7 días, Últimos 30 días, Anteriores.
export function groupByDate(conversations, now = new Date()) {
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000;
  const day = 24 * 3600;
  const groups = [
    { label: "Hoy", from: startOfToday },
    { label: "Ayer", from: startOfToday - day },
    { label: "Últimos 7 días", from: startOfToday - 7 * day },
    { label: "Últimos 30 días", from: startOfToday - 30 * day },
    { label: "Anteriores", from: -Infinity },
  ].map((group) => ({ ...group, items: [] }));
  for (const conversation of conversations) {
    groups.find((group) => conversation.updated_at >= group.from).items.push(conversation);
  }
  return groups.filter((group) => group.items.length > 0);
}
