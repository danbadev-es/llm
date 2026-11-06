import { useState } from "react";
import { groupByDate } from "../format.js";
import ConversationItem from "./ConversationItem.jsx";
import Icon from "./Icon.jsx";
import Modal from "./Modal.jsx";

function HistorySkeleton() {
  return (
    <div className="history-skeleton" aria-hidden="true">
      {[72, 55, 85, 62, 48].map((width, i) => (
        <span key={i} className="skeleton-line" style={{ width: `${width}%` }} />
      ))}
    </div>
  );
}

// The history column. On phones it becomes a drawer (see App: `inert` keeps the hidden one
// out of the tab order, and the backdrop closes it).
export default function Sidebar({
  open,
  hidden,
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

  return (
    <>
      <div className={`backdrop${open ? " visible" : ""}`} onClick={onClose} aria-hidden="true" />
      <aside className={`sidebar${open ? " open" : ""}`} inert={hidden} aria-label="Barra lateral">
        <div className="sidebar-top">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true">L</span>
            <span className="brand-name">Leon</span>
            <span className="brand-tag">local</span>
          </div>
          <button className="icon-button close-sidebar" type="button" aria-label="Cerrar menú" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>

        <button className="new-chat" type="button" title="Nueva conversación (Ctrl+Shift+O)" onClick={onNewChat}>
          <Icon name="newChat" size={17} />
          Nueva conversación
        </button>

        <nav className="history" aria-label="Historial de conversaciones" aria-busy={conversations === null}>
          {conversations === null ? (
            <HistorySkeleton />
          ) : conversations.length === 0 ? (
            <p className="history-empty">Tus conversaciones aparecerán aquí.</p>
          ) : (
            groupByDate(conversations).map((group, i) => (
              <section key={group.label} aria-labelledby={`history-group-${i}`}>
                <h3 id={`history-group-${i}`}>{group.label}</h3>
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
            ))
          )}
        </nav>

        <div className="sidebar-footer">
          <button className="sidebar-button" type="button" onClick={onOpenSettings}>
            <Icon name="sliders" size={17} />
            Ajustes
          </button>
        </div>
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
          <button type="button" className="button" data-autofocus onClick={() => setToDelete(null)}>
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
