import { useCallback, useEffect, useRef, useState } from "react";
import {
  createConversation,
  deleteConversation,
  fetchModels,
  getConversation,
  listConversations,
  renameConversation,
  streamGenerate,
  truncateConversation,
} from "./api.js";
import Composer from "./components/Composer.jsx";
import ContextMeter from "./components/ContextMeter.jsx";
import Header from "./components/Header.jsx";
import Icon from "./components/Icon.jsx";
import SettingsDialog from "./components/SettingsDialog.jsx";
import Sidebar from "./components/Sidebar.jsx";
import Thread from "./components/Thread.jsx";
import Toasts from "./components/Toasts.jsx";
import useContextUsage from "./hooks/useContextUsage.js";
import useMediaQuery from "./hooks/useMediaQuery.js";
import useToasts from "./hooks/useToasts.js";
import { loadMode, loadModelName, loadSettings, saveMode, saveModelName, saveSettings } from "./preferences.js";

// Must match the drawer breakpoint in styles.css.
const PHONE_QUERY = "(max-width: 760px)";

const isTyping = (element) => element?.closest?.("input, textarea, select, [contenteditable]");

// A message: {role: "user" | "assistant", content, prompt?, stats?, error?, streaming?}
// In "complete" mode the assistant message also carries the prompt it continued.

export default function App() {
  const [models, setModels] = useState([]);
  const [modelName, setModelName] = useState("");
  const [modelsState, setModelsState] = useState("loading"); // "loading" | "ready" | "error"
  const [conversations, setConversations] = useState(null); // null until the first list arrives
  const [currentId, setCurrentId] = useState(null);
  const [mode, setMode] = useState(loadMode);
  const [messages, setMessages] = useState([]);
  const [loadingConversation, setLoadingConversation] = useState(false);
  const [settings, setSettings] = useState(loadSettings);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState(""); // the composer's text, for the context meter
  // viewKey changes when another conversation replaces the one on screen; pinKey also on every send.
  // The thread re-pins its scroll to the bottom whenever pinKey changes.
  const [viewKey, setViewKey] = useState(0);
  const [pinKey, setPinKey] = useState(0);
  const { toasts, toast, dismiss } = useToasts();
  const isPhone = useMediaQuery(PHONE_QUERY);
  const composerRef = useRef(null);
  const controllerRef = useRef(null);
  // Each generation (and each conversation load) gets a number; leaving the conversation bumps it,
  // so a cancelled generation or a slow load can never write into the conversation that replaced it.
  const generationRef = useRef(0);

  const refreshConversations = useCallback(
    () =>
      listConversations()
        .then(setConversations)
        .catch(() => setConversations((current) => current ?? [])),
    [],
  );

  const loadModels = useCallback(() => {
    setModelsState("loading");
    fetchModels()
      .then((data) => {
        setModels(data.models);
        setModelsState("ready");
        // The model used last time, if it still exists; otherwise the first one.
        setModelName((current) => {
          const wanted = current || loadModelName();
          return (data.models.find((m) => m.name === wanted) ?? data.models[0])?.name ?? "";
        });
      })
      .catch(() => setModelsState("error"));
  }, []);

  useEffect(() => {
    loadModels();
    refreshConversations();
    if (window.matchMedia("(pointer: fine)").matches) composerRef.current?.focus();
  }, [loadModels, refreshConversations]);

  useEffect(() => saveSettings(settings), [settings]);
  useEffect(() => saveMode(mode), [mode]);
  useEffect(() => {
    if (modelName) saveModelName(modelName);
  }, [modelName]);

  const model = models.find((m) => m.name === modelName);
  const canSend = Boolean(model) && !loadingConversation;
  const contextUsage = useContextUsage({
    model,
    mode,
    system: settings.system,
    maxTokens: model ? Math.min(settings.maxTokens, model.max_new_tokens) : 0,
    messages,
    draft: draft.trim(),
    paused: busy || loadingConversation,
  });

  // Focus the composer, except on touch screens, where it would pop the keyboard up uninvited.
  function focusComposer({ force = false } = {}) {
    if (force || window.matchMedia("(pointer: fine)").matches) requestAnimationFrame(() => composerRef.current?.focus());
  }

  function updateLast(patch) {
    setMessages((current) => {
      if (current.length === 0) return current;
      const next = [...current];
      const last = next[next.length - 1];
      next[next.length - 1] = { ...last, ...(typeof patch === "function" ? patch(last) : patch) };
      return next;
    });
  }

  // Leave whatever is on screen: abort its generation and make its late updates stale.
  // busy is cleared right away so a new message can be sent at once; the aborted request
  // finishes in the background (the server still saves what had been written).
  function leaveConversation() {
    generationRef.current += 1;
    controllerRef.current?.abort();
    controllerRef.current = null;
    setBusy(false);
    setLoadingConversation(false);
    setViewKey((key) => key + 1);
    setPinKey((key) => key + 1);
  }

  function newChat() {
    leaveConversation();
    setCurrentId(null);
    setMessages([]);
    setSidebarOpen(false);
    focusComposer();
  }

  async function selectConversation(id) {
    setSidebarOpen(false);
    if (id === currentId && !loadingConversation) return;
    leaveConversation();
    const load = generationRef.current;
    setCurrentId(id); // highlight it in the history at once, while the messages load
    setMessages([]);
    setLoadingConversation(true);
    try {
      const conversation = await getConversation(id);
      if (load !== generationRef.current) return; // another conversation was opened meanwhile
      setMode(conversation.mode);
      if (models.some((m) => m.name === conversation.model)) setModelName(conversation.model);
      setMessages(conversation.messages);
      setPinKey((key) => key + 1);
      focusComposer();
    } catch (error) {
      if (load !== generationRef.current) return;
      toast(`No se pudo abrir la conversación: ${error.message}`, "error");
      setCurrentId(null);
      refreshConversations(); // it may have been deleted elsewhere
    } finally {
      if (load === generationRef.current) setLoadingConversation(false);
    }
  }

  function changeMode(nextMode) {
    if (nextMode === mode) return;
    setMode(nextMode);
    newChat(); // a conversation has a single mode: switching starts a new one
  }

  // Optimistic: the history changes at once; if the server disagrees, the list is reloaded.
  async function rename(id, title) {
    setConversations((list) => list.map((c) => (c.id === id ? { ...c, title } : c)));
    try {
      await renameConversation(id, title);
      toast("Conversación renombrada");
    } catch (error) {
      toast(`No se pudo renombrar: ${error.message}`, "error");
    }
    refreshConversations(); // the server normalizes titles (one line, at most 60 characters)
  }

  async function remove(id) {
    setConversations((list) => list.filter((c) => c.id !== id));
    if (id === currentId) newChat();
    try {
      await deleteConversation(id);
      toast("Conversación eliminada");
    } catch (error) {
      toast(`No se pudo eliminar: ${error.message}`, "error");
      refreshConversations();
    }
  }

  async function copy(text) {
    try {
      await navigator.clipboard.writeText(text);
      toast("Copiado al portapapeles");
      return true;
    } catch {
      toast("No se pudo copiar al portapapeles", "error");
      return false;
    }
  }

  // Returns true if the message was sent (the composer only clears its text then).
  function send(text) {
    text = text.trim();
    if (!text || !canSend || busy) return false;
    generate(text, messages);
    return true;
  }

  // Regenerate the last answer, or (with newText) edit the last user message / prompt and resend.
  // The thread keeps everything before that last exchange and replaces the exchange itself, on
  // screen and in the saved history (generate deletes the old attempt there first).
  function redoLast(newText) {
    if (!canSend || busy) return;
    if (mode === "chat") {
      const index = messages.findLastIndex((m) => m.role === "user");
      if (index >= 0) generate(newText ?? messages[index].content, messages.slice(0, index), { replace: true });
    } else {
      const last = messages.at(-1);
      if (last?.prompt !== undefined) generate(newText ?? last.prompt, messages.slice(0, -1), { replace: true });
    }
  }

  // Streams one answer after `base` (the messages kept before the new exchange). With replace,
  // the saved history is first cut back to `base` too, so a replaced attempt never reappears.
  async function generate(text, base, { replace = false } = {}) {
    const generation = ++generationRef.current;
    const patch = (change) => generation === generationRef.current && updateLast(change);
    const body = {
      model: model.name,
      mode,
      temperature: settings.temperature,
      top_k: settings.topK,
      max_new_tokens: Math.min(settings.maxTokens, model.max_new_tokens),
    };
    const answer = { role: "assistant", content: "", streaming: true };
    if (mode === "chat") {
      // Answers that failed or produced nothing are not part of the conversation.
      const history = base
        .filter((m) => m.role === "user" || (!m.error && m.content))
        .map(({ role, content }) => ({ role, content }));
      body.messages = [...history, { role: "user", content: text }];
      body.system = settings.system;
      setMessages([...base, { role: "user", content: text }, answer]);
    } else {
      body.prompt = text;
      setMessages([...base, { ...answer, prompt: text }]);
    }
    setPinKey((key) => key + 1); // sending always brings the view back to the newest text

    // Tokens can arrive faster than the screen refreshes: they are buffered and painted at most
    // once per frame, so a fast model costs one React render per frame, not one per token.
    let pending = "";
    let frame = 0;
    const flush = () => {
      cancelAnimationFrame(frame);
      frame = 0;
      if (!pending) return;
      const chunk = pending;
      pending = "";
      patch((last) => ({ content: last.content + chunk }));
    };

    const controller = new AbortController();
    controllerRef.current = controller;
    setBusy(true);
    let finished = false;
    try {
      // The first message of a new chat creates its conversation; the server saves every exchange.
      let conversationId = currentId;
      if (replace && conversationId) await truncateConversation(conversationId, base.length); // saved = thread, 1:1
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
              pending += event.text;
              frame ||= requestAnimationFrame(flush);
            } else if (event.done) {
              finished = true;
              flush();
              patch({ stats: event });
            } else if (event.error) {
              throw new Error(event.error);
            }
          },
        },
      );
      if (!finished) throw new Error("la conexión con la API se cortó");
    } catch (error) {
      flush();
      if (error.name === "AbortError") patch((last) => ({ stats: last.stats ?? { stopped: "user" } }));
      else patch({ error: error.message });
    } finally {
      patch({ streaming: false });
      // Only the generation still in charge may clear busy: an aborted one finishing late
      // must not unlock the composer while a newer answer is streaming.
      if (controllerRef.current === controller) {
        controllerRef.current = null;
        setBusy(false);
      }
      refreshConversations(); // the conversation moves to the top of the history
    }
  }

  const stop = () => controllerRef.current?.abort();

  // Global shortcuts. The handler is registered once and reads the latest state through a ref.
  const shortcuts = useRef(null);
  shortcuts.current = { newChat, stop, busy, sidebarOpen, closeSidebar: () => setSidebarOpen(false), focusComposer };
  useEffect(() => {
    function onKeyDown(e) {
      const s = shortcuts.current;
      if (document.querySelector("dialog[open]")) return; // the dialog owns the keyboard
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "o") {
        e.preventDefault();
        s.newChat();
        s.focusComposer({ force: true });
      } else if (e.key === "Escape" && e.shiftKey) {
        e.preventDefault();
        s.focusComposer({ force: true });
      } else if (e.key === "Escape") {
        if (s.busy) s.stop();
        else if (s.sidebarOpen) s.closeSidebar();
      } else if (e.key === "/" && !isTyping(e.target) && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault();
        s.focusComposer({ force: true });
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const notice =
    modelsState === "error" ? (
      <div className="notice error" role="alert">
        <Icon name="alert" size={18} />
        <p>No se pudo contactar con la API. ¿Está corriendo apps/api/app.py?</p>
        <button type="button" className="button small" onClick={loadModels}>
          Reintentar
        </button>
      </div>
    ) : mode === "chat" && model?.stage === "base" && !loadingConversation ? (
      <div className="notice">
        <Icon name="alert" size={18} />
        <p>
          <b>{model.name}</b> es un modelo base: solo sabe continuar texto. El formato de chat lo aprenderá en la
          fase 5; hasta entonces sus respuestas no tendrán sentido.
        </p>
        <button type="button" className="button small" onClick={() => changeMode("complete")}>
          Usar Completar
        </button>
      </div>
    ) : null;

  return (
    <div className="app">
      <Sidebar
        open={sidebarOpen}
        hidden={isPhone && !sidebarOpen}
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
      {/* With the drawer open on a phone, the page behind it is inert: no focus, no clicks. */}
      <main className="main" inert={isPhone && sidebarOpen}>
        <Header
          models={models}
          model={model}
          modelsState={modelsState}
          onModelChange={setModelName}
          mode={mode}
          onModeChange={changeMode}
          onOpenSidebar={() => setSidebarOpen(true)}
          onNewChat={newChat}
        />
        <Thread
          mode={mode}
          messages={messages}
          loading={loadingConversation}
          busy={busy}
          canSend={canSend}
          viewKey={viewKey}
          pinKey={pinKey}
          notice={notice}
          onSuggestion={send}
          onCopy={copy}
          onRegenerate={() => redoLast()}
          onEdit={redoLast}
        />
        <Composer
          inputRef={composerRef}
          mode={mode}
          busy={busy}
          disabled={!canSend}
          onSend={send}
          onStop={stop}
          onTextChange={setDraft}
          meter={<ContextMeter usage={contextUsage} mode={mode} />}
        />
      </main>
      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onChange={setSettings}
        maxTokensLimit={model?.max_new_tokens ?? 512}
      />
      <Toasts toasts={toasts} onDismiss={dismiss} />
    </div>
  );
}
