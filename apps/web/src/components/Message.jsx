import { useEffect, useRef, useState } from "react";
import { formatStats } from "../format.js";
import { isSendKey, useAutoGrow } from "./Composer.jsx";
import Icon from "./Icon.jsx";

function ActionButton({ icon, label, onClick }) {
  return (
    <button type="button" className="icon-button action" aria-label={label} title={label} onClick={onClick}>
      <Icon name={icon} size={16} />
    </button>
  );
}

// Copy with a short check mark on the button itself; App also confirms it with a toast.
function CopyButton({ text, onCopy }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(timer);
  }, [copied]);
  return (
    <ActionButton
      icon={copied ? "check" : "copy"}
      label={copied ? "Copiado" : "Copiar"}
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
    <div className="message-editor">
      <textarea
        ref={ref}
        rows={1}
        value={text}
        aria-label="Editar mensaje"
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
      <div className="message-editor-actions">
        <button type="button" className="button" onClick={onCancel}>
          Cancelar
        </button>
        <button type="button" className="button primary" disabled={!text.trim()} onClick={submit}>
          Enviar
        </button>
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <span className="typing" role="status" aria-label="Leon está escribiendo">
      <span />
      <span />
      <span />
    </span>
  );
}

// A message: {role: "user" | "assistant", content, prompt?, stats?, error?, streaming?}.
// In Completar mode the assistant message carries the prompt it continued: the prompt is shown
// highlighted and the continuation as plain text, so it is always clear which words are the model's.
export default function Message({ message, canEdit, canRegenerate, onCopy, onRegenerate, onEdit }) {
  const [editing, setEditing] = useState(false);
  const { role, content, prompt, error, streaming } = message;
  const isCompletion = prompt !== undefined;

  if (editing) {
    return (
      <div className={`message ${role}`}>
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

  const editButton = canEdit && (
    <ActionButton icon="pencil" label={isCompletion ? "Editar texto" : "Editar mensaje"} onClick={() => setEditing(true)} />
  );

  if (role === "user") {
    return (
      <article className="message user" aria-label="Tu mensaje">
        <div className="user-bubble">{content}</div>
        <div className="message-actions">{editButton}</div>
      </article>
    );
  }

  const stats = formatStats(message.stats);
  return (
    <article className="message assistant" aria-label="Respuesta de Leon" aria-busy={streaming || undefined}>
      <div className="answer">
        {isCompletion && (
          <mark className="prompt" title="Tu texto">
            <span className="visually-hidden">Tu texto: </span>
            {prompt}
          </mark>
        )}
        {isCompletion && content && <span className="visually-hidden"> Continuación: </span>}
        {content}
        {streaming && (content || isCompletion ? <span className="caret" aria-hidden="true" /> : <TypingDots />)}
      </div>
      {error && (
        <div className="message-error" role="alert">
          <Icon name="alert" size={17} />
          <span className="message-error-text">{error}</span>
          {canRegenerate && (
            <button type="button" className="button small" onClick={onRegenerate}>
              <Icon name="retry" size={15} />
              Reintentar
            </button>
          )}
        </div>
      )}
      {/* The row has a fixed height from the first token, so the buttons appearing at the end
          of the stream never push the layout. */}
      <div className="message-actions">
        {!streaming && (
          <>
            {(content || prompt) && <CopyButton text={(prompt ?? "") + content} onCopy={onCopy} />}
            {canRegenerate && <ActionButton icon="retry" label="Regenerar" onClick={onRegenerate} />}
            {editButton}
          </>
        )}
        {stats && <span className="message-stats">{stats}</span>}
      </div>
    </article>
  );
}
