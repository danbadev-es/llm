import { useEffect, useId, useRef } from "react";
import Icon from "./Icon.jsx";

// A modal dialog on top of the native <dialog> element: it brings focus trapping, Esc to close,
// the dimmed backdrop and focus restoration on close for free. Clicking the backdrop closes it too.
// The element marked data-autofocus (if any) gets the focus when it opens, instead of the ✕ button.
export default function Modal({ open, onClose, title, children, className = "" }) {
  const ref = useRef(null);
  const titleId = useId();

  useEffect(() => {
    const dialog = ref.current;
    if (open && !dialog.open) {
      dialog.showModal();
      dialog.querySelector("[data-autofocus]")?.focus();
    }
    if (!open && dialog.open) dialog.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      className={`modal ${className}`}
      aria-labelledby={titleId}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()} // a click outside the panel lands on <dialog>
    >
      {/* Rendering the content only while open keeps the inactive dialog out of the DOM's tab
          order and resets its fields' local state every time it opens. */}
      {open && (
        <div className="modal-panel">
          <div className="modal-header">
            <h2 id={titleId}>{title}</h2>
            <button className="icon-button" type="button" aria-label="Cerrar" onClick={onClose}>
              <Icon name="close" />
            </button>
          </div>
          {children}
        </div>
      )}
    </dialog>
  );
}
