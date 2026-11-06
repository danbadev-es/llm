import Modal from "./Modal.jsx";

export const DEFAULT_SETTINGS = { temperature: 0.8, topK: 40, maxTokens: 200, system: "" };

function Slider({ id, label, hint, value, display, min, max, step, onChange }) {
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
      <p className="hint-text">{hint}</p>
    </div>
  );
}

export default function SettingsDialog({ open, onClose, settings, onChange, maxTokensLimit }) {
  const set = (key) => (value) => onChange({ ...settings, [key]: value });

  return (
    <Modal open={open} onClose={onClose} title="Ajustes">
      <div className="settings-body">
        <div className="field">
          <label htmlFor="system">Instrucciones del sistema</label>
          <textarea
            id="system"
            value={settings.system}
            placeholder="Eres Leon, un asistente que responde en español de forma breve."
            onChange={(e) => set("system")(e.target.value)}
          />
          <p className="hint-text">Solo en modo Chat: se envían al principio de cada conversación.</p>
        </div>
        <Slider
          id="temperature"
          label="Temperatura"
          hint="Más baja: respuestas más seguras y repetitivas. Más alta: más variadas y arriesgadas."
          value={settings.temperature}
          min={0}
          max={1.5}
          step={0.05}
          display={settings.temperature === 0 ? "0 (greedy)" : settings.temperature.toFixed(2)}
          onChange={set("temperature")}
        />
        <Slider
          id="top-k"
          label="Top-k"
          hint="Solo se elige entre los k tokens más probables. 0 = sin límite."
          value={settings.topK}
          min={0}
          max={200}
          step={5}
          display={settings.topK === 0 ? "sin límite" : settings.topK}
          onChange={set("topK")}
        />
        <Slider
          id="max-tokens"
          label="Tokens máximos"
          hint={`Longitud máxima de cada respuesta (hasta ${maxTokensLimit} con este modelo).`}
          value={Math.min(settings.maxTokens, maxTokensLimit)}
          min={8}
          max={maxTokensLimit}
          step={8}
          display={Math.min(settings.maxTokens, maxTokensLimit)}
          onChange={set("maxTokens")}
        />
      </div>
      <div className="modal-actions">
        <button type="button" className="button" onClick={() => onChange(DEFAULT_SETTINGS)}>
          Restablecer
        </button>
        <button type="button" className="button primary" onClick={onClose}>
          Listo
        </button>
      </div>
    </Modal>
  );
}
