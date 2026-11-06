import { useRef, useState } from "react";
import Icon from "./Icon.jsx";
import Menu from "./Menu.jsx";

const MODE_LABEL = { complete: "Completar", chat: "Chat" };

// One row of the history: the title opens it; the ⋯ menu renames it in place or deletes it.
export default function ConversationItem({ conversation, active, onSelect, onRename, onAskDelete }) {
  const [editing, setEditing] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const titleRef = useRef(null);
  // Esc unmounts the focused field, which fires one last blur: this flag makes that blur a no-op.
  const finished = useRef(false);

  function startRename() {
    finished.current = false;
    setEditing(true);
  }

  function finishRename(input, save) {
    if (finished.current) return;
    finished.current = true;
    const clean = input.value.trim();
    setEditing(false);
    if (save && clean && clean !== conversation.title) onRename(conversation.id, clean);
    // Give the focus back to the row, where the keyboard user started.
    requestAnimationFrame(() => titleRef.current?.focus());
  }

  return (
    <li className={`conversation${active ? " active" : ""}${menuOpen ? " menu-open" : ""}`}>
      {editing ? (
        <input
          className="rename-input"
          defaultValue={conversation.title}
          autoFocus
          aria-label="Nuevo nombre de la conversación"
          maxLength={120}
          onFocus={(e) => e.target.select()}
          onBlur={(e) => finishRename(e.target, true)}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.target.blur(); // blur saves, so Enter and clicking away agree
            if (e.key === "Escape") {
              e.stopPropagation(); // Esc cancels the rename only
              finishRename(e.target, false);
            }
          }}
        />
      ) : (
        <>
          <button
            ref={titleRef}
            type="button"
            className="conversation-title"
            title={conversation.title}
            aria-current={active ? "page" : undefined}
            onClick={() => onSelect(conversation.id)}
          >
            <Icon name={conversation.mode === "chat" ? "chat" : "text"} size={15} className="mode-icon" />
            <span className="visually-hidden">{MODE_LABEL[conversation.mode]}: </span>
            <span className="conversation-text">{conversation.title}</span>
          </button>
          <Menu
            label={`Opciones de «${conversation.title}»`}
            className="options"
            onOpenChange={setMenuOpen}
            items={[
              { label: "Renombrar", icon: "pencil", keepFocus: true, onSelect: startRename },
              { label: "Eliminar", icon: "trash", danger: true, onSelect: () => onAskDelete(conversation) },
            ]}
          />
        </>
      )}
    </li>
  );
}
