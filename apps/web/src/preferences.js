// Per-browser preferences in localStorage: generation settings, the last model and the last mode.
// They are a convenience: if storage is blocked (private mode, strict settings), the defaults still work.

export const DEFAULT_SETTINGS = { temperature: 0.8, topK: 40, maxTokens: 200, system: "" };

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
    return { ...DEFAULT_SETTINGS, ...JSON.parse(read(KEYS.settings) ?? "{}") };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export const saveSettings = (settings) => write(KEYS.settings, JSON.stringify(settings));
export const loadModelName = () => read(KEYS.model);
export const saveModelName = (name) => write(KEYS.model, name);
export const loadMode = () => (read(KEYS.mode) === "chat" ? "chat" : "complete");
export const saveMode = (mode) => write(KEYS.mode, mode);
