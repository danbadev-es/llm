// Per-browser preferences in localStorage: generation settings, the last model and the last mode.
// They are a convenience: if storage is blocked (private mode, strict settings), the defaults still work.

// The system prompt Leon saw in a share of its SFT conversations (leon/chat.py DEFAULT_SYSTEM).
export const DEFAULT_SYSTEM = "Eres Leon, un asistente que responde en español de forma clara y breve.";
// version 2: the system prompt defaults to DEFAULT_SYSTEM (version 1 saved "" for "none" by default).
// version 3: temperature 0.5 instead of 0.8: a 56M model drifts into nonsense when it samples
// unlikely words, and below ~0.4 it starts repeating itself.
// tools: Leon's own tools (calculator, date, document search) in chat mode, for phase 6 models.
export const DEFAULT_SETTINGS = {
  temperature: 0.5,
  topK: 40,
  maxTokens: 200,
  system: DEFAULT_SYSTEM,
  tools: true,
  version: 3,
};

const KEYS = { settings: "leon.settings", model: "leon.model", mode: "leon.mode" };

function read(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // the preference just will not persist
  }
}

export function loadSettings() {
  try {
    const saved = JSON.parse(read(KEYS.settings) ?? "{}");
    // Settings saved before version 2 hold system "" only because that was the default.
    if (!saved.version && saved.system === "") delete saved.system;
    // ...and before version 3, temperature 0.8 only because that was the default.
    if ((saved.version ?? 0) < 3 && saved.temperature === 0.8) delete saved.temperature;
    return { ...DEFAULT_SETTINGS, ...saved, version: DEFAULT_SETTINGS.version };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export const saveSettings = (settings) => write(KEYS.settings, JSON.stringify(settings));
export const loadModelName = () => read(KEYS.model);
export const saveModelName = (name) => write(KEYS.model, name);
// null when nothing was saved: the app then picks the mode that suits the model it opens with.
export const loadMode = () => {
  const mode = read(KEYS.mode);
  return mode === "chat" || mode === "complete" ? mode : null;
};
export const saveMode = (mode) => write(KEYS.mode, mode);
