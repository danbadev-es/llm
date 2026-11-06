// Client for apps/api. Every call goes to /api/... on the same origin
// (Flask in production, proxied by Vite in development).

async function request(method, url, body) {
  const response = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    let message = `${method} ${url}: ${response.status}`;
    try {
      message = (await response.json()).error ?? message;
    } catch {
      // not a JSON error body
    }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

export const fetchModels = () => request("GET", "/api/models");

// --- conversation history ---
export const listConversations = () => request("GET", "/api/conversations").then((data) => data.conversations);
export const getConversation = (id) => request("GET", `/api/conversations/${id}`);
export const createConversation = (mode, model, title) =>
  request("POST", "/api/conversations", { mode, model, title });
export const renameConversation = (id, title) => request("PATCH", `/api/conversations/${id}`, { title });
export const deleteConversation = (id) => request("DELETE", `/api/conversations/${id}`);

// POST /api/generate answers with Server-Sent Events: "data: {json}\n\n" per event.
// onEvent receives {text}, then a final {done, tokens, tok_per_s, ...} or an {error}.
// With body.conversation_id, the server also saves the exchange in that conversation.
export async function streamGenerate(body, { signal, onEvent }) {
  const response = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      message = (await response.json()).error ?? message;
    } catch {
      // not a JSON error body
    }
    throw new Error(message);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop(); // the last piece may be an incomplete event
    for (const raw of events) {
      if (raw.startsWith("data: ")) onEvent(JSON.parse(raw.slice(6)));
    }
  }
}
