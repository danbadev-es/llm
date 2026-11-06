// Client for apps/api. Every call goes to /api/... on the same origin
// (Flask in production, proxied by Vite in development).

export async function fetchModels() {
  const response = await fetch("/api/models");
  if (!response.ok) throw new Error(`GET /api/models: ${response.status}`);
  return response.json();
}

// POST /api/generate answers with Server-Sent Events: "data: {json}\n\n" per event.
// onEvent receives {text}, then a final {done, tokens, tok_per_s, ...} or an {error}.
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
