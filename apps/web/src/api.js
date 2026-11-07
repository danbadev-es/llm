// Client for apps/api, on the same origin (Flask in production, proxied by Vite in development).
//
// Generation and models use the Anthropic-compatible API (/v1/...), the same one any Anthropic
// client can use. The conversation history uses the app's own REST endpoints (/api/conversations).
// This module translates between those wire formats and the small shapes the components use,
// so the rest of the app does not depend on either of them.

const CONVERSATION_HEADER = "X-Leon-Conversation";

// The message of an error response: Anthropic's envelope {error: {message}} or the app's {error}.
async function errorMessage(response, fallback) {
  try {
    const body = await response.json();
    return body.error?.message ?? body.error ?? fallback;
  } catch {
    return fallback; // not a JSON body
  }
}

async function request(method, url, body) {
  const response = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) throw new Error(await errorMessage(response, `${method} ${url}: ${response.status}`));
  return response.status === 204 ? null : response.json();
}

// GET /v1/models → [{name, params, context, max_new_tokens, step, stage}] (Leon's extras live under "leon").
export async function fetchModels() {
  const { data } = await request("GET", "/v1/models");
  return {
    models: data.map((model) => ({
      name: model.id,
      params: model.leon.params,
      context: model.leon.context,
      max_new_tokens: model.leon.max_output_tokens,
      step: model.leon.step,
      stage: model.leon.stage,
    })),
  };
}

// --- conversation history ---
export const listConversations = () => request("GET", "/api/conversations").then((data) => data.conversations);
export const getConversation = (id) => request("GET", `/api/conversations/${id}`);
export const createConversation = (mode, model, title) =>
  request("POST", "/api/conversations", { mode, model, title });
export const renameConversation = (id, title) => request("PATCH", `/api/conversations/${id}`, { title });
export const deleteConversation = (id) => request("DELETE", `/api/conversations/${id}`);
// Keep the first `keep` saved messages: regenerating or editing replaces the last exchange.
export const truncateConversation = (id, keep) =>
  request("DELETE", `/api/conversations/${id}/messages?from=${keep}`);

// The app's generation request → an Anthropic request: chat mode is the Messages API,
// complete mode (raw continuation) the legacy Text Completions API.
function toAnthropic(body) {
  const sampling = { temperature: body.temperature, top_k: body.top_k, stream: true };
  if (body.mode === "chat") {
    return {
      url: "/v1/messages",
      payload: {
        model: body.model,
        max_tokens: body.max_new_tokens,
        messages: body.messages,
        ...(body.system?.trim() ? { system: body.system } : {}),
        // Leon's own tools, run on the server (phase 6): the answer streams their calls and results.
        ...(body.tools ? { leon: { tools: true } } : {}),
        ...sampling,
      },
    };
  }
  return {
    url: "/v1/complete",
    payload: { model: body.model, prompt: body.prompt, max_tokens_to_sample: body.max_new_tokens, ...sampling },
  };
}

// Anthropic stream events → the app's events: {text}, {tool_call}, {tool_result}, {done, ...stats}
// or {error}. Stateful: a tool call's input arrives as JSON pieces (input_json_delta) between the
// start and the stop of its block, so one parser lives as long as one answer.
function anthropicParser() {
  const calls = new Map(); // block index → {id, name, json}
  return (event) => {
    if (event.type === "content_block_start" && event.content_block?.type === "server_tool_use") {
      const { id, name } = event.content_block;
      calls.set(event.index, { id, name, json: "" });
      return null;
    }
    if (event.type === "content_block_start" && event.content_block?.type === "leon_tool_result") {
      return { tool_result: { id: event.content_block.tool_use_id, content: event.content_block.content } };
    }
    if (event.type === "content_block_delta" && event.delta?.type === "input_json_delta") {
      const call = calls.get(event.index);
      if (call) call.json += event.delta.partial_json;
      return null;
    }
    if (event.type === "content_block_stop" && calls.has(event.index)) {
      const { id, name, json } = calls.get(event.index);
      calls.delete(event.index);
      let input = {};
      try {
        input = JSON.parse(json || "{}");
      } catch {
        input = { raw: json };
      }
      return { tool_call: { id, name, input } };
    }
    return fromAnthropic(event);
  };
}

// One stateless Anthropic stream event → the app's events: {text}, {done, ...stats} or {error}.
function fromAnthropic(event) {
  switch (event.type) {
    case "content_block_delta":
      return event.delta?.type === "text_delta" ? { text: event.delta.text } : null;
    case "message_delta":
      return { done: true, ...event.leon }; // Leon's generation stats ride on the final delta
    case "completion":
      // Legacy completions: every event carries text; the last one also has stop_reason and stats.
      return event.stop_reason ? { text: event.completion, final: { done: true, ...event.leon } } : { text: event.completion };
    case "error":
      return { error: event.error?.message ?? "error del servidor" };
    default:
      return null; // message_start, content_block_start/stop, message_stop, ping
  }
}

// Stream one answer. onEvent receives {text} pieces ({tool_call} and {tool_result} with tools), then
// {done, tokens, tok_per_s, prompt_tokens, dropped_turns, stopped} or {error}.
// With body.conversation_id the server saves the exchange.
export async function streamGenerate(body, { signal, onEvent }) {
  const { url, payload } = toAnthropic(body);
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(body.conversation_id ? { [CONVERSATION_HEADER]: body.conversation_id } : {}),
    },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) throw new Error(await errorMessage(response, response.statusText));

  // Server-Sent Events: "event: <type>" and "data: <json>" lines, events separated by a blank line.
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parse = anthropicParser();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop(); // the last piece may be an incomplete event
    for (const raw of events) {
      const data = raw.split("\n").find((line) => line.startsWith("data: "));
      if (!data) continue;
      const event = parse(JSON.parse(data.slice(6)));
      if (!event) continue;
      if (event.final) {
        if (event.text) onEvent({ text: event.text });
        onEvent(event.final);
      } else {
        onEvent(event);
      }
    }
  }
}

// POST /api/context → how the next request would fill the model's context (the composer's meter):
// {context, reserved, system, history, draft, used, free, total, dropped_messages, dropped_tokens}.
export const fetchContextUsage = (body, signal) =>
  fetch("/api/context", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  }).then((response) => (response.ok ? response.json() : Promise.reject(new Error(String(response.status)))));
