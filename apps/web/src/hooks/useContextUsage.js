import { useEffect, useState } from "react";
import { fetchContextUsage } from "../api.js";

const DEBOUNCE_MS = 250; // while typing, ask at most once per pause

// The context breakdown of the request that sending now would make: system instructions,
// history (the same messages generate() sends) and the draft in the composer. Not refreshed
// while an answer streams (paused); it catches up as soon as the answer is complete.
export default function useContextUsage({ model, mode, system, tools, maxTokens, messages, draft, paused }) {
  const [usage, setUsage] = useState(null);
  const history =
    mode === "chat"
      ? messages
          .filter((m) => !m.streaming && (m.role === "user" || (!m.error && m.content)))
          .map(({ role, content }) => ({ role, content }))
      : [];
  const request = model
    ? JSON.stringify({
        model: model.name,
        mode,
        system: mode === "chat" ? system : "",
        tools: mode === "chat" && tools,
        max_tokens: maxTokens,
        messages: history,
        draft,
      })
    : "";

  useEffect(() => {
    if (!request || paused) return undefined;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      fetchContextUsage(JSON.parse(request), controller.signal)
        .then(setUsage)
        .catch(() => {}); // the meter is a hint: if the API is down, it keeps its last value
    }, DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [request, paused]);

  return model ? usage : null;
}
