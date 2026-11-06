// Top of the conversation: menu button (phones), model picker and mode, like ChatGPT's model selector.
export default function Header({ models, model, loadError, onModelChange, mode, onModeChange, onOpenSidebar }) {
  return (
    <header className="topbar">
      <button className="icon-button open-sidebar" aria-label="Abrir menú" onClick={onOpenSidebar}>
        ☰
      </button>
      <select
        className="model-picker"
        aria-label="Modelo"
        value={model?.name ?? ""}
        onChange={(e) => onModelChange(e.target.value)}
        title={model ? `${(model.params / 1e6).toFixed(1)}M parámetros · contexto ${model.context} · paso ${model.step}` : ""}
      >
        {models.length === 0 && <option value="">{loadError || "Sin modelos"}</option>}
        {models.map((m) => (
          <option key={m.name} value={m.name}>
            {m.name}
            {m.stage === "base" ? " (base)" : ""}
          </option>
        ))}
      </select>
      <div className="segmented" role="group" aria-label="Modo">
        {[
          ["complete", "Completar"],
          ["chat", "Chat"],
        ].map(([value, text]) => (
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
    </header>
  );
}
