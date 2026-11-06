import { ArrowUpIcon, SquareIcon } from "lucide-react";
import { useEffect, useLayoutEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { WithTooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

const MAX_HEIGHT = 220; // px; past this the text area scrolls instead of growing

// Grows a <textarea> with its content. Shared with the inline message editor.
export function useAutoGrow(ref, value, maxHeight = MAX_HEIGHT) {
  useLayoutEffect(() => {
    const input = ref.current;
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, maxHeight)}px`;
  }, [ref, value, maxHeight]);
}

// Enter sends, Shift+Enter adds a line. isComposing: Enter that confirms an IME composition
// (accents on some keyboards, Japanese input…) must not send the message.
export const isSendKey = (e) => e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing;

// The input at the bottom. inputRef is owned by App, so global shortcuts can focus it.
// onTextChange lets App measure the draft (the context meter); meter is rendered by the send button.
export default function Composer({ inputRef, mode, busy, disabled, onSend, onStop, onTextChange, meter }) {
  const [text, setText] = useState("");
  useAutoGrow(inputRef, text);
  useEffect(() => onTextChange?.(text), [text, onTextChange]);
  const empty = text.trim() === "";

  function submit() {
    if (busy) return onStop();
    if (onSend(text)) setText(""); // keep the text if it could not be sent
  }

  return (
    <div className="mx-auto w-full max-w-[47rem] shrink-0 px-3 pb-2 md:px-6 md:pb-3">
      <form
        className="flex items-end gap-2 rounded-[22px] border border-input bg-background py-2 pr-2 pl-4 shadow-md transition-[border-color,box-shadow] focus-within:border-primary focus-within:ring-[3px] focus-within:ring-primary/15"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <textarea
          ref={inputRef}
          rows={1}
          value={text}
          aria-label={mode === "chat" ? "Mensaje" : "Comienzo del texto"}
          placeholder={mode === "chat" ? "Escribe un mensaje…" : "Escribe el comienzo de un texto…"}
          className="min-h-[38px] min-w-0 flex-1 resize-none bg-transparent py-[7px] text-base leading-normal outline-none placeholder:text-muted-foreground"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (!isSendKey(e)) return;
            e.preventDefault();
            if (!busy && !empty) submit(); // while generating, Enter never doubles as "stop"
          }}
        />
        {meter}
        <WithTooltip label={busy ? "Detener (Esc)" : "Enviar (Enter)"}>
          <Button
            type="submit"
            size="icon"
            disabled={!busy && (disabled || empty)}
            aria-label={busy ? "Detener (Esc)" : "Enviar"}
            className={cn(
              "size-[38px] rounded-full active:scale-95 disabled:bg-input disabled:text-background disabled:opacity-100",
              busy && "bg-foreground text-background hover:bg-foreground/90",
            )}
          >
            {busy ? <SquareIcon className="size-3.5 fill-current" /> : <ArrowUpIcon className="size-[18px]" />}
          </Button>
        </WithTooltip>
      </form>
      <p className="mt-2 hidden text-center text-xs text-muted-foreground md:block">
        {mode === "chat" ? "Enter para enviar · Shift+Enter para nueva línea" : "Leon continúa tu texto · Enter para enviar · Shift+Enter para nueva línea"}
      </p>
    </div>
  );
}
