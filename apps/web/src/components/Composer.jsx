import { useLayoutEffect, useRef, useState } from "react";

export default function Composer({ mode, busy, disabled, onSend, onStop }) {
  const [text, setText] = useState("");
  const inputRef = useRef(null);

  useLayoutEffect(() => {
    const input = inputRef.current; // grow with the content, up to 200px
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
  }, [text]);

  function submit() {
    if (busy) return onStop();
    if (onSend(text)) setText(""); // keep the text if it could not be sent
  }

  return (
    <div className="composer">
      <div className="column">
        <form onSubmit={(e) => { e.preventDefault(); submit(); }}>
          <textarea
            ref={inputRef}
            rows={1}
            value={text}
            autoFocus
            aria-label={mode === "chat" ? "Mensaje" : "Comienzo del texto"}
            placeholder={mode === "chat" ? "Escribe un mensaje…" : "Escribe el comienzo de un texto…"}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (!busy) submit();
              }
            }}
          />
          <button className="send" type="submit" disabled={disabled && !busy} aria-label={busy ? "Detener" : "Enviar"} title={busy ? "Detener" : "Enviar"}>
            {busy ? "■" : "↑"}
          </button>
        </form>
        <div className="hint">
          {mode === "chat" ? "Enter para enviar · Shift+Enter para salto de línea" : "El modelo continúa tu texto · Enter para enviar"}
        </div>
      </div>
    </div>
  );
}
