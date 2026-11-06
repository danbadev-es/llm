import { useEffect, useRef, useState } from "react";
import { fetchModels, streamGenerate } from "./api.js";
import Composer from "./components/Composer.jsx";
import Sidebar from "./components/Sidebar.jsx";
import Thread from "./components/Thread.jsx";

const DEFAULT_SETTINGS = { temperature: 0.8, topK: 40, maxTokens: 200, system: "" };

// A message: {role: "user" | "assistant", content, prompt?, meta?, error?, streaming?}
// In "complete" mode the assistant message also carries the prompt it continued.

export default function App() {
  const [models, setModels] = useState([]);
  const [modelName, setModelName] = useState("");
  const [loadError, setLoadError] = useState("");
  const [mode, setMode] = useState("complete");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const controllerRef = useRef(null);

  useEffect(() => {
    fetchModels()
      .then((data) => {
        setModels(data.models);
        if (data.models.length) setModelName(data.models[0].name);
      })
      .catch(() => setLoadError("No se pudo contactar con la API. ¿Está corriendo apps/api/app.py?"));
  }, []);

  const model = models.find((m) => m.name === modelName);

  // Keep "max tokens" within what the selected model allows (half of its context).
  useEffect(() => {
    if (model && settings.maxTokens > model.max_new_tokens) {
      setSettings((s) => ({ ...s, maxTokens: model.max_new_tokens }));
    }
  }, [model, settings.maxTokens]);

  function updateLast(patch) {
    setMessages((current) => {
      if (current.length === 0) return current; // the chat was cleared while generating
      const next = [...current];
      const last = next[next.length - 1];
      next[next.length - 1] = { ...last, ...(typeof patch === "function" ? patch(last) : patch) };
      return next;
    });
  }

  function newChat() {
    controllerRef.current?.abort();
    setMessages([]);
  }

  function changeMode(nextMode) {
    setMode(nextMode);
    newChat();
  }

  // Returns true if the message was sent (the composer only clears its text then).
  function send(text) {
    text = text.trim();
    if (!text || !model || busy) return false;
    generate(text);
    return true;
  }

  async function generate(text) {
    const body = {
      model: model.name,
      mode,
      temperature: settings.temperature,
      top_k: settings.topK,
      max_new_tokens: settings.maxTokens,
    };
    if (mode === "chat") {
      // Answers that failed or produced nothing are not part of the conversation.
      const history = messages
        .filter((m) => m.role === "user" || (!m.error && m.content))
        .map(({ role, content }) => ({ role, content }));
      body.messages = [...history, { role: "user", content: text }];
      body.system = settings.system;
      setMessages((current) => [
        ...current,
        { role: "user", content: text },
        { role: "assistant", content: "", streaming: true },
      ]);
    } else {
      body.prompt = text;
      setMessages((current) => [...current, { role: "assistant", prompt: text, content: "", streaming: true }]);
    }

    const controller = new AbortController();
    controllerRef.current = controller;
    setBusy(true);
    let finished = false;
    try {
      await streamGenerate(body, {
        signal: controller.signal,
        onEvent: (event) => {
          if (event.text !== undefined) {
            updateLast((last) => ({ content: last.content + event.text }));
          } else if (event.done) {
            finished = true;
            const limit = event.stopped === "max_tokens" ? " · límite de tokens" : "";
            const dropped = event.dropped_turns ? ` · ${event.dropped_turns} turnos antiguos omitidos` : "";
            updateLast({
              meta: `${event.tokens} tokens · ${event.tok_per_s.toFixed(0)} tok/s · contexto ${event.prompt_tokens}${limit}${dropped}`,
            });
          } else if (event.error) {
            throw new Error(event.error);
          }
        },
      });
      if (!finished) throw new Error("la conexión con la API se cortó");
    } catch (error) {
      if (error.name === "AbortError") updateLast((last) => ({ meta: last.meta ?? "Detenido" }));
      else updateLast({ error: error.message });
    } finally {
      updateLast({ streaming: false });
      controllerRef.current = null;
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <Sidebar
        models={models}
        model={model}
        loadError={loadError}
        onModelChange={setModelName}
        mode={mode}
        onModeChange={changeMode}
        settings={settings}
        onSettingsChange={setSettings}
        onNewChat={newChat}
      />
      <main>
        <Thread mode={mode} messages={messages} onSuggestion={send} />
        <Composer
          mode={mode}
          busy={busy}
          disabled={!model}
          onSend={send}
          onStop={() => controllerRef.current?.abort()}
        />
      </main>
    </div>
  );
}
