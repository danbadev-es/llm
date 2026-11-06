import { useEffect, useState } from "react";

const MOBILE_QUERY = "(max-width: 760px)";

function Slider({ id, label, value, display, min, max, step, onChange }) {
  return (
    <div className="field">
      <div className="row">
        <label htmlFor={id}>{label}</label>
        <span className="value">{display}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

export default function Sidebar({
  models, model, loadError, onModelChange, mode, onModeChange, settings, onSettingsChange, onNewChat,
}) {
  // On phones the settings start folded; on desktop they are always visible.
  const [settingsOpen, setSettingsOpen] = useState(() => !window.matchMedia(MOBILE_QUERY).matches);
  useEffect(() => {
    const media = window.matchMedia(MOBILE_QUERY);
    const sync = () => !media.matches && setSettingsOpen(true);
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);

  const set = (key) => (value) => onSettingsChange({ ...settings, [key]: value });

  return (
    <aside>
      <div className="brand">
        <h1>Leon</h1>
        <span>0.1 · local</span>
      </div>
      <button className="new-chat" onClick={onNewChat}>＋ Nueva conversación</button>

      <div className="field">
        <label htmlFor="model">Modelo</label>
        <select id="model" value={model?.name ?? ""} onChange={(e) => onModelChange(e.target.value)}>
          {models.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name}{m.stage === "base" ? " (base)" : ""}
            </option>
          ))}
        </select>
        <div className="model-info">
          {loadError ||
            (model
              ? `${(model.params / 1e6).toFixed(1)}M parámetros · contexto ${model.context} · paso ${model.step}`
              : models.length === 0 && "No hay checkpoints. Entrena uno con phases/3_modern/train.py.")}
        </div>
      </div>

      <div className="field">
        <span className="label" id="mode-label">Modo</span>
        <div className="segmented" role="group" aria-labelledby="mode-label">
          {[["complete", "Completar"], ["chat", "Chat"]].map(([value, text]) => (
            <button
              key={value}
              type="button"
              className={mode === value ? "active" : ""}
              aria-pressed={mode === value}
              onClick={() => onModeChange(value)}
            >
              {text}
            </button>
          ))}
        </div>
      </div>
      {mode === "chat" && model?.stage === "base" && (
        <div className="notice">
          Este es un modelo <b>base</b>: solo sabe continuar texto. El modo Chat le envía el formato{" "}
          <code>&lt;|user|&gt;…&lt;|assistant|&gt;</code>, que aprenderá en la fase 5. Hasta entonces, las respuestas
          no tendrán sentido.
        </div>
      )}

      <details open={settingsOpen} onToggle={(e) => setSettingsOpen(e.currentTarget.open)}>
        <summary>Ajustes</summary>
        <div className="settings-body">
          {mode === "chat" && (
            <div className="field">
              <label htmlFor="system">Instrucciones del sistema</label>
              <textarea
                id="system"
                value={settings.system}
                placeholder="Eres Leon, un asistente que responde en español de forma breve."
                onChange={(e) => set("system")(e.target.value)}
              />
            </div>
          )}
          <Slider
            id="temperature"
            label="Temperatura" value={settings.temperature} min={0} max={1.5} step={0.05}
            display={settings.temperature === 0 ? "0 (greedy)" : settings.temperature.toFixed(2)}
            onChange={set("temperature")}
          />
          <Slider
            id="top-k"
            label="Top-k" value={settings.topK} min={0} max={200} step={5}
            display={settings.topK === 0 ? "sin límite" : settings.topK}
            onChange={set("topK")}
          />
          <Slider
            id="max-tokens"
            label="Tokens máximos" value={settings.maxTokens} min={8} max={model?.max_new_tokens ?? 512} step={8}
            display={settings.maxTokens}
            onChange={set("maxTokens")}
          />
        </div>
      </details>
    </aside>
  );
}
