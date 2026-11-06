import { useCallback, useState } from "react";

const DURATION_MS = 4000;
const MAX_VISIBLE = 3;
let nextId = 1;

// A minimal toast queue: toast("Copiado") or toast("Algo falló", "error").
// Toasts dismiss themselves; the newest ones win when there are too many.
export default function useToasts() {
  const [toasts, setToasts] = useState([]);

  const dismiss = useCallback((id) => setToasts((list) => list.filter((t) => t.id !== id)), []);

  const toast = useCallback(
    (message, tone = "info") => {
      const id = nextId++;
      setToasts((list) => [...list.slice(-(MAX_VISIBLE - 1)), { id, message, tone }]);
      setTimeout(() => dismiss(id), tone === "error" ? DURATION_MS * 1.5 : DURATION_MS);
    },
    [dismiss],
  );

  return { toasts, toast, dismiss };
}
