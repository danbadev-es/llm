import { CheckIcon, CircleAlertIcon, CopyIcon, PencilIcon, RotateCcwIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import WithTooltip from "./WithTooltip.jsx";
import { cn } from "cn";
import { formatStats } from "../format.js";
import { isSendKey, useAutoGrow } from "./Composer.jsx";

// Spacing between messages: an answer gets room from its question, consecutive continuations
// (Completar) are separate texts and get a bit more.
const GAP = { "user>assistant": "mt-4", "assistant>assistant": "mt-5" };

function ActionButton({ icon: IconComponent, label, onClick, visible }) {
  return (
    <WithTooltip label={label}>
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label={label}
        onClick={onClick}
        className={cn(
          "size-7.5 text-muted-foreground opacity-0 transition-opacity group-hover/message:opacity-100 group-focus-within/message:opacity-100 hover:text-foreground [@media(hover:none)]:opacity-100",
          visible && "opacity-100",
        )}
      >
        <IconComponent />
      </Button>
    </WithTooltip>
  );
}

// Copy with a short check mark on the button itself; App also confirms it with a toast.
function CopyButton({ text, onCopy, visible }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return undefined;
    const timer = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(timer);
  }, [copied]);
  return (
    <ActionButton
      icon={copied ? CheckIcon : CopyIcon}
      label={copied ? "Copiado" : "Copiar"}
      visible={visible}
      onClick={async () => setCopied(await onCopy(text))}
    />
  );
}

// Edits the last user message (Chat) or the last prompt (Completar) and sends it again.
function MessageEditor({ initial, onCancel, onSubmit }) {
  const [text, setText] = useState(initial);
  const ref = useRef(null);
  useAutoGrow(ref, text, 320);
  useEffect(() => {
    const input = ref.current;
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }, []);
  const submit = () => text.trim() && onSubmit(text.trim());

  return (
    <div className="w-full rounded-2xl border border-input bg-background p-3 shadow-xs transition-colors focus-within:border-primary">
      <textarea
        ref={ref}
        rows={1}
        value={text}
        aria-label="Editar mensaje"
        className="block w-full resize-none bg-transparent px-2 py-1 leading-[1.7] outline-none"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (isSendKey(e)) {
            e.preventDefault();
            submit();
          } else if (e.key === "Escape") {
            e.stopPropagation(); // Esc cancels the edit only
            onCancel();
          }
        }}
      />
      <div className="mt-2 flex justify-end gap-2">
        <Button variant="outline" size="sm" onClick={onCancel}>
          Cancelar
        </Button>
        <Button size="sm" disabled={!text.trim()} onClick={submit}>
          Enviar
        </Button>
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <span className="inline-flex h-[1.7em] items-center gap-1.5" role="status" aria-label="Leon está escribiendo">
      {[0, 150, 300].map((delay) => (
        <span key={delay} className="size-1.5 animate-soft-pulse rounded-full bg-muted-foreground" style={{ animationDelay: `${delay}ms` }} />
      ))}
    </span>
  );
}

// A message: {role: "user" | "assistant", content, prompt?, stats?, error?, streaming?}.
// In Completar mode the assistant message carries the prompt it continued: the prompt is shown
// highlighted and the continuation as plain text, so it is always clear which words are the model's.
export default function Message({ message, previousRole, isLast, canEdit, canRegenerate, onCopy, onRegenerate, onEdit }) {
  const [editing, setEditing] = useState(false);
  const { role, content, prompt, error, streaming } = message;
  const isCompletion = prompt !== undefined;
  const spacing = previousRole ? (GAP[`${previousRole}>${role}`] ?? "mt-2") : "";
  const base = cn("group/message flex animate-fade-up flex-col text-base leading-[1.7]", spacing);

  if (editing) {
    return (
      <div className={base}>
        <MessageEditor
          initial={isCompletion ? prompt : content}
          onCancel={() => setEditing(false)}
          onSubmit={(text) => {
            setEditing(false);
            onEdit(text);
          }}
        />
      </div>
    );
  }

  // The last answer keeps its actions visible; the others show them on hover or focus.
  const visible = isLast && role === "assistant";
  const editButton = canEdit && (
    <ActionButton
      icon={PencilIcon}
      label={isCompletion ? "Editar texto" : "Editar mensaje"}
      visible={visible}
      onClick={() => setEditing(true)}
    />
  );

  if (role === "user") {
    return (
      <article className={cn(base, "items-end")} aria-label="Tu mensaje">
        <div className="max-w-[min(85%,36rem)] rounded-[22px] bg-bubble px-4 py-2.5 break-words whitespace-pre-wrap">{content}</div>
        <div className="mt-1 flex min-h-8 items-center gap-0.5">{editButton}</div>
      </article>
    );
  }

  const stats = formatStats(message.stats);
  return (
    <article className={base} aria-label="Respuesta de Leon" aria-busy={streaming || undefined}>
      <div className="break-words whitespace-pre-wrap">
        {isCompletion && (
          // box-decoration-break: the highlight wraps line by line, like a marker pen.
          <mark className="-mx-px rounded px-[3px] py-0.5 [box-decoration-break:clone] bg-prompt text-prompt-foreground" title="Tu texto">
            <span className="sr-only">Tu texto: </span>
            {prompt}
          </mark>
        )}
        {isCompletion && content && <span className="sr-only"> Continuación: </span>}
        {content}
        {streaming &&
          (content || isCompletion ? (
            <span className="ml-[3px] inline-block size-[0.5em] animate-soft-pulse rounded-full bg-foreground align-[0.05em]" aria-hidden="true" />
          ) : (
            <TypingDots />
          ))}
      </div>
      {error && (
        <div
          className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-xl border border-danger-border bg-danger-soft px-4 py-3 text-sm leading-normal text-destructive"
          role="alert"
        >
          <CircleAlertIcon className="size-[17px] shrink-0" />
          <span className="min-w-48 flex-1">{error}</span>
          {canRegenerate && (
            <Button variant="outline" size="sm" className="border-current bg-transparent text-inherit hover:bg-destructive/10" onClick={onRegenerate}>
              <RotateCcwIcon />
              Reintentar
            </Button>
          )}
        </div>
      )}
      {/* Fixed height from the first token: the buttons appearing at the end never push the layout. */}
      <div className="mt-1 -ml-1.5 flex min-h-8 flex-wrap items-center gap-0.5 text-xs">
        {!streaming && (
          <>
            {(content || prompt) && <CopyButton text={(prompt ?? "") + content} onCopy={onCopy} visible={visible} />}
            {canRegenerate && <ActionButton icon={RotateCcwIcon} label="Regenerar" visible={visible} onClick={onRegenerate} />}
            {editButton}
          </>
        )}
        {stats && <span className="ml-auto pl-3 text-muted-foreground tabular-nums">{stats}</span>}
      </div>
    </article>
  );
}
