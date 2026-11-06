import { useEffect, useRef, useState } from "react";
import { groupByDate } from "../format.js";
import Modal from "./Modal.jsx";

function ConversationItem({ conversation, active, onSelect, onRename, onAskDelete }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(conversation.title);
  const itemRef = useRef(null);

  // Close the options menu on any click outside this item.
  useEffect(() => {
    if (!menuOpen) return;
    const close = (e) => !itemRef.current?.contains(e.target) && setMenuOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [menuOpen]);

  function finishRename(save) {
    setEditing(false);
    const clean = title.trim();
    if (save && clean && clean !== conversation.title) onRename(conversation.id, clean);
    else setTitle(conversation.title);
  }

  return (
    <li ref={itemRef} className={`conversation ${active ? "active" : ""} ${menuOpen ? "menu-open" : ""}`}>
      {editing ? (
        <input
          className="rename-input"
          value={title}
          autoFocus
          aria-label="Nuevo nombre"
          onChange={(e) => setTitle(e.target.value)}
          onFocus={(e) => e.target.select()}
          onBlur={() => finishRename(true)}
          onKeyDown={(e) => {
            if (e.key === "Enter") finishRename(true);
            if (e.key === "Escape") finishRename(false);
          }}
        />
      ) : (
        <button
          className="conversation-title"
          title={conversation.title}
          aria-current={active ? "page" : undefined}
          onClick={() => onSelect(conversation.id)}
        >
          {conversation.mode === "complete" && (
            <span className="mode-tag" title="Modo completar">
              ✎
            </span>
          )}
          {conversation.title}
        </button>
      )}
      {!editing && (
        <button
          className="icon-button options"
          aria-label={`Opciones de «${conversation.title}»`}
          aria-haspopup="menu"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((open) => !open)}
        >
          ⋯
        </button>
      )}
      {menuOpen && (
        <div className="menu" role="menu">
          <button
            role="menuitem"
            onClick={() => {
              setMenuOpen(false);
              setTitle(conversation.title);
              setEditing(true);
            }}
          >
            ✎ Renombrar
          </button>
          <button
            role="menuitem"
            className="danger"
            onClick={() => {
              setMenuOpen(false);
              onAskDelete(conversation);
            }}
          >
            🗑 Eliminar
          </button>
        </div>
      )}
    </li>
  );
}

export default function Sidebar({
  open,
  onClose,
  conversations,
  currentId,
  onSelect,
  onNewChat,
  onRename,
  onDelete,
  onOpenSettings,
}) {
  const [toDelete, setToDelete] = useState(null);
  const groups = groupByDate(conversations);

  return (
    <>
      <div className={`backdrop ${open ? "visible" : ""}`} onClick={onClose} />
      <aside className={open ? "open" : ""}>
        <div className="brand">
          <h1>Leon</h1>
          <span>0.1 · local</span>
          <button className="icon-button close-sidebar" aria-label="Cerrar menú" onClick={onClose}>
            ✕
          </button>
        </div>
        <button className="new-chat" onClick={onNewChat}>
          ＋ Nueva conversación
        </button>

        <nav className="history" aria-label="Historial de conversaciones">
          {groups.length === 0 && <p className="empty">Tus conversaciones aparecerán aquí.</p>}
          {groups.map((group) => (
            <section key={group.label}>
              <h3>{group.label}</h3>
              <ul>
                {group.items.map((conversation) => (
                  <ConversationItem
                    key={conversation.id}
                    conversation={conversation}
                    active={conversation.id === currentId}
                    onSelect={onSelect}
                    onRename={onRename}
                    onAskDelete={setToDelete}
                  />
                ))}
              </ul>
            </section>
          ))}
        </nav>

        <button className="settings-button" onClick={onOpenSettings}>
          ⚙ Ajustes
        </button>
      </aside>

      <Modal
        open={toDelete !== null}
        onClose={() => setToDelete(null)}
        title="¿Eliminar conversación?"
        className="small"
      >
        <p className="modal-text">
          Se borrará <b>{toDelete?.title}</b> con todos sus mensajes. No se puede deshacer.
        </p>
        <div className="modal-actions">
          <button type="button" className="button" onClick={() => setToDelete(null)}>
            Cancelar
          </button>
          <button
            type="button"
            className="button danger"
            onClick={() => {
              onDelete(toDelete.id);
              setToDelete(null);
            }}
          >
            Eliminar
          </button>
        </div>
      </Modal>
    </>
  );
}
