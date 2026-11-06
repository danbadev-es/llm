import { useCallback, useEffect, useRef, useState } from "react";
import {
  createConversation,
  deleteConversation,
  fetchModels,
  getConversation,
  listConversations,
  renameConversation,
  streamGenerate,
} from "./api.js";
import Composer from "./components/Composer.jsx";
import Header from "./components/Header.jsx";
import SettingsDialog, { DEFAULT_SETTINGS } from "./components/SettingsDialog.jsx";
import Sidebar from "./components/Sidebar.jsx";
import Thread from "./components/Thread.jsx";

const SETTINGS_KEY = "leon.settings";
const MODEL_KEY = "leon.model";

function storageGet(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function storageSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // private mode or blocked storage: the preference just will not persist
  }
}

// Settings are a per-browser convenience: if storage is unavailable, the defaults still work.
function loadSettings() {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(storageGet(SETTINGS_KEY) ?? "{}") };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

// A message: {role: "user" | "assistant", content, prompt?, stats?, error?, streaming?}
// In "complete" mode the assistant message also carries the prompt it continued.

export default function App() {
  const [models, setModels] = useState([]);
  const [modelName, setModelName] = useState("");
  const [loadError, setLoadError] = useState("");
  const [conversations, setConversations] = useState([]);
  const [currentId, setCurrentId] = useState(null);
  const [mode, setMode] = useState("complete");
  const [messages, setMessages] = useState([]);
  const [settings, setSettings] = useState(loadSettings);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const controllerRef = useRef(null);
  // Each generation gets a number; leaving the conversation bumps it, so a cancelled
  // generation can never write into the conversation that replaced it on screen.
  const generationRef = useRef(0);

  const refreshConversations = useCallback(() => listConversations().then(setConversations).catch(() => {}), []);

  useEffect(() => {
    fetchModels()
      .then((data) => {
        setModels(data.models);
        // The model used last time, if it still exists; otherwise the first one.
        const remembered = storageGet(MODEL_KEY);
        const initial = data.models.find((m) => m.name === remembered) ?? data.models[0];
        if (initial) setModelName(initial.name);
      })
      .catch(() => setLoadError("No se pudo contactar con la API. ¿Está corriendo apps/api/app.py?"));
    refreshConversations();
  }, [refreshConversations]);

  useEffect(() => storageSet(SETTINGS_KEY, JSON.stringify(settings)), [settings]);
  useEffect(() => {
    if (modelName) storageSet(MODEL_KEY, modelName);
  }, [modelName]);

  const model = models.find((m) => m.name === modelName);

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
    generationRef.current += 1;
    controllerRef.current?.abort();
    setCurrentId(null);
    setMessages([]);
    setSidebarOpen(false);
  }

  async function selectConversation(id) {
    generationRef.current += 1;
    controllerRef.current?.abort();
    setSidebarOpen(false);
    try {
      const conversation = await getConversation(id);
      setCurrentId(id);
      setMode(conversation.mode);
      if (models.some((m) => m.name === conversation.model)) setModelName(conversation.model);
      setMessages(conversation.messages);
    } catch (error) {
      setLoadError(error.message);
      refreshConversations(); // it may have been deleted elsewhere
    }
  }

  function changeMode(nextMode) {
    if (nextMode === mode) return;
    setMode(nextMode);
    newChat(); // a conversation has a single mode: switching starts a new one
  }

  async function rename(id, title) {
    await renameConversation(id, title).catch(() => {});
    refreshConversations();
  }

  async function remove(id) {
    await deleteConversation(id).catch(() => {});
    if (id === currentId) newChat();
    refreshConversations();
  }

  // Returns true if the message was sent (the composer only clears its text then).
  function send(text) {
    text = text.trim();
    if (!text || !model || busy) return false;
    generate(text);
    return true;
  }

  async function generate(text) {
    const generation = ++generationRef.current;
    const patch = (change) => generation === generationRef.current && updateLast(change);
    const body = {
      model: model.name,
      mode,
      temperature: settings.temperature,
      top_k: settings.topK,
      max_new_tokens: Math.min(settings.maxTokens, model.max_new_tokens),
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
      // The first message of a new chat creates its conversation; the server saves every exchange.
      let conversationId = currentId;
      if (!conversationId) {
        conversationId = (await createConversation(mode, model.name, text)).id;
        if (generation === generationRef.current) setCurrentId(conversationId);
        refreshConversations();
      }
      await streamGenerate(
        { ...body, conversation_id: conversationId },
        {
          signal: controller.signal,
          onEvent: (event) => {
            if (event.text !== undefined) {
              patch((last) => ({ content: last.content + event.text }));
            } else if (event.done) {
              finished = true;
              patch({ stats: event });
            } else if (event.error) {
              throw new Error(event.error);
            }
          },
        },
      );
      if (!finished) throw new Error("la conexión con la API se cortó");
    } catch (error) {
      if (error.name === "AbortError") patch((last) => ({ stats: last.stats ?? { stopped: "user" } }));
      else patch({ error: error.message });
    } finally {
      patch({ streaming: false });
      if (controllerRef.current === controller) controllerRef.current = null;
      setBusy(false);
      refreshConversations(); // the conversation moves to the top of the history
    }
  }

  const showBaseNotice = mode === "chat" && model?.stage === "base";

  return (
    <div className="app">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        conversations={conversations}
        currentId={currentId}
        onSelect={selectConversation}
        onNewChat={newChat}
        onRename={rename}
        onDelete={remove}
        onOpenSettings={() => {
          setSidebarOpen(false);
          setSettingsOpen(true);
        }}
      />
      <main>
        <Header
          models={models}
          model={model}
          loadError={loadError}
          onModelChange={setModelName}
          mode={mode}
          onModeChange={changeMode}
          onOpenSidebar={() => setSidebarOpen(true)}
        />
        {showBaseNotice && (
          <div className="notice">
            <b>{model.name}</b> es un modelo base: solo sabe continuar texto. El formato de chat lo aprenderá en la
            fase 5; hasta entonces sus respuestas no tendrán sentido. Prueba el modo <b>Completar</b>.
          </div>
        )}
        <Thread mode={mode} messages={messages} onSuggestion={send} />
        <Composer
          mode={mode}
          busy={busy}
          disabled={!model}
          onSend={send}
          onStop={() => controllerRef.current?.abort()}
        />
      </main>
      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onChange={setSettings}
        maxTokensLimit={model?.max_new_tokens ?? 512}
      />
    </div>
  );
}
