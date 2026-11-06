import { useEffect, useId, useRef, useState } from "react";

const RADIUS = 8;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
const number = (n) => n.toLocaleString("es-ES");
const percent = (part, whole) => Math.round((100 * part) / whole);

// What happens when the context is full, in the words of this request.
function note(usage, mode) {
  const { dropped_tokens: droppedTokens, dropped_messages: droppedMessages } = usage;
  if (droppedTokens === 0) {
    return mode === "chat"
      ? "Cuando se llene, Leon dejará de ver los mensajes más antiguos, que seguirán guardados en el historial."
      : "Si el texto no cabe, Leon solo lee su final.";
  }
  if (mode !== "chat" || droppedMessages === 0) {
    return `${mode === "chat" ? "El mensaje" : "El texto"} no cabe entero: Leon solo leerá sus últimos ${number(usage.draft)} tokens.`;
  }
  const which = droppedMessages === 1 ? "el mensaje más antiguo" : `los ${droppedMessages} mensajes más antiguos`;
  return `No cabe todo: Leon dejará de ver ${which} (${number(droppedTokens)} tokens). Siguen guardados en el historial.`;
}

// The context ring in the composer, like Claude's: how full the model's context window will be
// when this message is sent. Hover, focus or tap shows the breakdown.
export default function ContextMeter({ usage, mode }) {
  const [pinned, setPinned] = useState(false); // opened by a click or tap, closed by another
  const ref = useRef(null);
  const popoverId = useId();

  useEffect(() => {
    if (!pinned) return undefined;
    const close = (e) => {
      if (!ref.current?.contains(e.target) || e.key === "Escape") setPinned(false);
    };
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("pointerdown", close);
      document.removeEventListener("keydown", close);
    };
  }, [pinned]);

  if (!usage) return null;
  const { context, reserved, used, free } = usage;
  const filled = used + reserved; // the answer's room is taken as soon as the request is sent
  const share = filled / context;
  const level = usage.dropped_tokens > 0 ? "full" : share >= 0.9 ? "high" : share >= 0.7 ? "warn" : "ok";
  const rows =
    mode === "chat"
      ? [
          { key: "system", label: "Instrucciones del sistema", value: usage.system },
          { key: "history", label: "Conversación", value: usage.history },
          { key: "draft", label: "Mensaje nuevo", value: usage.draft },
        ]
      : [{ key: "draft", label: "Texto a continuar", value: usage.draft }];
  rows.push({ key: "reserved", label: "Reservado para la respuesta", value: reserved }, { key: "free", label: "Libre", value: free });

  return (
    <div className={`context-meter ${level}${pinned ? " pinned" : ""}`} ref={ref}>
      <button
        type="button"
        className="context-ring"
        aria-label={`Contexto: ${percent(filled, context)} % usado. Ver desglose`}
        aria-expanded={pinned}
        aria-controls={popoverId}
        onClick={() => setPinned((open) => !open)}
      >
        <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
          <circle className="ring-track" cx="11" cy="11" r={RADIUS} />
          <circle
            className="ring-fill"
            cx="11"
            cy="11"
            r={RADIUS}
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={CIRCUMFERENCE * (1 - Math.min(1, share))}
          />
        </svg>
      </button>
      <div className="context-popover" id={popoverId} role="tooltip">
        <p className="context-title">
          Contexto <span>{percent(filled, context)} %</span>
        </p>
        <p className="context-subtitle">
          {number(filled)} de {number(context)} tokens
        </p>
        <div className="context-bar" aria-hidden="true">
          {rows.map((row) =>
            row.value > 0 ? <span key={row.key} className={`segment ctx-${row.key}`} style={{ flexGrow: row.value }} /> : null,
          )}
        </div>
        <ul className="context-rows">
          {rows.map((row) => (
            <li key={row.key}>
              <span className={`swatch ctx-${row.key}`} aria-hidden="true" />
              <span className="label">{row.label}</span>
              <span className="value">{number(row.value)}</span>
            </li>
          ))}
        </ul>
        <p className="context-note">
          {note(usage, mode)}
        </p>
      </div>
    </div>
  );
}
