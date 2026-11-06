import { useRef, useState } from "react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";

const RADIUS = 8;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
const HOVER_CLOSE_MS = 150; // lets the pointer travel from the ring to the card without closing it
const number = (n) => n.toLocaleString("es-ES");
const percent = (part, whole) => Math.round((100 * part) / whole);

const RING = { ok: "stroke-muted-foreground", warn: "stroke-warn", high: "stroke-destructive", full: "stroke-destructive" };
const SWATCH = {
  system: "bg-ctx-system",
  history: "bg-ctx-history",
  draft: "bg-ctx-draft",
  reserved: "bg-ctx-reserved",
  free: "bg-transparent",
};

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
// when this message is sent. Hovering shows the breakdown; a click or tap keeps it open.
export default function ContextMeter({ usage, mode }) {
  const [open, setOpen] = useState(false);
  const closeTimer = useRef(0);
  const hover = {
    onPointerEnter: (e) => {
      if (e.pointerType !== "mouse") return; // touch: the tap (onClick) toggles it
      clearTimeout(closeTimer.current);
      setOpen(true);
    },
    onPointerLeave: (e) => {
      if (e.pointerType !== "mouse") return;
      closeTimer.current = setTimeout(() => setOpen(false), HOVER_CLOSE_MS);
    },
  };

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
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild {...hover}>
        <button
          type="button"
          className="mb-[3px] grid size-8 shrink-0 place-items-center rounded-full transition-colors outline-none hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring/50 data-[state=open]:bg-accent"
          aria-label={`Contexto: ${percent(filled, context)} % usado. Ver desglose`}
        >
          {/* Rotated so the arc starts at 12 o'clock and fills clockwise. */}
          <svg width="22" height="22" viewBox="0 0 22 22" className="-rotate-90" aria-hidden="true">
            <circle cx="11" cy="11" r={RADIUS} fill="none" strokeWidth="2.5" className="stroke-border" />
            <circle
              cx="11"
              cy="11"
              r={RADIUS}
              fill="none"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeDasharray={CIRCUMFERENCE}
              strokeDashoffset={CIRCUMFERENCE * (1 - Math.min(1, share))}
              className={cn("transition-[stroke-dashoffset,stroke] duration-300 ease-(--ease-out)", RING[level])}
            />
          </svg>
        </button>
      </PopoverTrigger>
      <PopoverContent
        side="top"
        align="end"
        alignOffset={-46}
        sideOffset={12}
        className="w-[284px] px-4 pt-3 pb-4 text-[13px]"
        onOpenAutoFocus={(e) => e.preventDefault()} // the focus stays in the composer
        {...hover}
      >
        <p className="flex justify-between font-semibold">
          Contexto <span>{percent(filled, context)} %</span>
        </p>
        <p className="mt-0.5 mb-3 text-muted-foreground">
          {number(filled)} de {number(context)} tokens
        </p>
        <div className="mb-3 flex h-2 gap-0.5 overflow-hidden rounded-full bg-accent" aria-hidden="true">
          {rows.map((row) =>
            row.value > 0 ? <span key={row.key} className={cn("min-w-[3px] basis-0", SWATCH[row.key])} style={{ flexGrow: row.value }} /> : null,
          )}
        </div>
        <ul className="grid gap-1.5">
          {rows.map((row) => (
            <li key={row.key} className="flex items-center gap-2 text-secondary-text">
              <span
                className={cn("size-2.5 shrink-0 rounded-[3px]", SWATCH[row.key], row.key === "free" && "ring-1 ring-input ring-inset")}
                aria-hidden="true"
              />
              <span className="flex-1">{row.label}</span>
              <span className="text-foreground tabular-nums">{number(row.value)}</span>
            </li>
          ))}
        </ul>
        <p className={cn("mt-3 border-t pt-3 leading-normal text-muted-foreground", level === "full" && "text-destructive")}>
          {note(usage, mode)}
        </p>
      </PopoverContent>
    </Popover>
  );
}
