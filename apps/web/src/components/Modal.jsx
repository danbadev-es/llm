import { useEffect, useRef } from "react";

// A modal dialog on top of the native <dialog> element: it brings focus trapping,
// Esc to close and the dimmed backdrop for free. Clicking the backdrop closes it too.
export default function Modal({ open, onClose, title, children, className = "" }) {
  const ref = useRef(null);

  useEffect(() => {
    const dialog = ref.current;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      className={`modal ${className}`}
      aria-label={title}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()} // a click outside the panel lands on <dialog>
    >
      <div className="modal-panel">
        <div className="modal-header">
          <h2>{title}</h2>
          <button className="icon-button" type="button" aria-label="Cerrar" onClick={onClose}>
            ✕
          </button>
        </div>
        {children}
      </div>
    </dialog>
  );
}
