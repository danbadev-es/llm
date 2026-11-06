import Icon from "./Icon.jsx";

const MODES = [
  ["complete", "Completar", "El modelo continúa tu texto"],
  ["chat", "Chat", "Conversación por turnos"],
];

// "3_modern/wiki_32m" → group "3_modern", option "wiki_32m": the closed picker then shows the
// short, distinctive part of the name, which still fits on a 360 px phone.
function groupModels(models) {
  const groups = new Map();
  for (const model of models) {
    const slash = model.name.lastIndexOf("/");
    const group = slash === -1 ? "" : model.name.slice(0, slash);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push({ ...model, label: model.name.slice(slash + 1) + (model.stage === "base" ? " (base)" : "") });
  }
  return [...groups];
}

function modelTitle(model) {
  return model ? `${model.name} · ${(model.params / 1e6).toFixed(1)}M parámetros · contexto ${model.context} · paso ${model.step}` : "";
}

// Top of the conversation: menu and new-chat buttons (phones), the model picker and the mode.
// The picker is a styled native <select>: it is keyboard and screen-reader friendly everywhere,
// and on phones it opens the system's own picker.
export default function Header({ models, model, modelsState, onModelChange, mode, onModeChange, onOpenSidebar, onNewChat }) {
  const placeholder = { loading: "Cargando modelos…", error: "Sin conexión", ready: "Sin modelos" }[modelsState];

  return (
    <header className="topbar">
      <button className="icon-button mobile-only" type="button" aria-label="Abrir menú" onClick={onOpenSidebar}>
        <Icon name="menu" size={20} />
      </button>

      <div className="model-picker" title={modelTitle(model)}>
        <select
          aria-label="Modelo"
          value={model?.name ?? ""}
          disabled={models.length === 0}
          onChange={(e) => onModelChange(e.target.value)}
        >
          {models.length === 0 && <option value="">{placeholder}</option>}
          {groupModels(models).map(([group, items]) => (
            <optgroup key={group} label={group || "Modelos"}>
              {items.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.label}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        <Icon name="chevronDown" size={16} className="model-picker-chevron" />
      </div>

      <div className="segmented" role="group" aria-label="Modo">
        {MODES.map(([value, text, description]) => (
          <button
            key={value}
            type="button"
            title={description}
            aria-pressed={mode === value}
            onClick={() => onModeChange(value)}
          >
            {text}
          </button>
        ))}
      </div>

      <button className="icon-button mobile-only" type="button" aria-label="Nueva conversación" onClick={onNewChat}>
        <Icon name="newChat" size={19} />
      </button>
    </header>
  );
}
