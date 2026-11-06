import { DEFAULT_SETTINGS } from "../preferences.js";
import Modal from "./Modal.jsx";

const SHORTCUTS = [
  [["Ctrl", "Shift", "O"], "Nueva conversación"],
  [["/"], "Escribir en el cuadro de texto"],
  [["Enter"], "Enviar"],
  [["Shift", "Enter"], "Nueva línea"],
  [["Esc"], "Detener la respuesta o cerrar"],
];

function Slider({ id, label, hint, value, display, min, max, step, onChange }) {
  return (
    <div className="field">
      <div className="field-row">
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id} className="field-value">
          {display}
        </output>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-describedby={`${id}-hint`}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <p id={`${id}-hint`} className="field-hint">
        {hint}
      </p>
    </div>
  );
}

// Changes apply (and persist) immediately: "Listo" only closes the dialog.
export default function SettingsDialog({ open, onClose, settings, onChange, maxTokensLimit }) {
  const set = (key) => (value) => onChange({ ...settings, [key]: value });
  const maxTokens = Math.min(settings.maxTokens, maxTokensLimit);

  return (
    <Modal open={open} onClose={onClose} title="Ajustes" className="settings">
      <div className="settings-body">
        <div className="field">
          <label htmlFor="system">Instrucciones del sistema</label>
          <textarea
            id="system"
            rows={3}
            value={settings.system}
            placeholder="Eres Leon, un asistente que responde en español de forma breve."
            aria-describedby="system-hint"
            onChange={(e) => set("system")(e.target.value)}
          />
          <p id="system-hint" className="field-hint">
            Solo en modo Chat: se envían al principio de cada conversación.
          </p>
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
          value={maxTokens}
          min={8}
          max={maxTokensLimit}
          step={8}
          display={maxTokens}
          onChange={set("maxTokens")}
        />

        <section className="shortcuts" aria-labelledby="shortcuts-title">
          <h3 id="shortcuts-title">Atajos de teclado</h3>
          <dl>
            {SHORTCUTS.map(([keys, action]) => (
              <div key={action}>
                <dt>{action}</dt>
                <dd>
                  {keys.map((key) => (
                    <kbd key={key}>{key}</kbd>
                  ))}
                </dd>
              </div>
            ))}
          </dl>
        </section>
      </div>
      <div className="modal-actions">
        <button type="button" className="button" onClick={() => onChange(DEFAULT_SETTINGS)}>
          Restablecer
        </button>
        <button type="button" className="button primary" data-autofocus onClick={onClose}>
          Listo
        </button>
      </div>
    </Modal>
  );
}
