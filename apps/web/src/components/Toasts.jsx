import Icon from "./Icon.jsx";

// The toast stack (bottom of the screen). The live region is always in the DOM: screen readers
// only announce changes to a region that already existed, not a region that appears with its text.
export default function Toasts({ toasts, onDismiss }) {
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast ${toast.tone}`} role={toast.tone === "error" ? "alert" : "status"}>
          <Icon name={toast.tone === "error" ? "alert" : "check"} size={16} />
          <span>{toast.message}</span>
          <button type="button" className="icon-button" aria-label="Cerrar aviso" onClick={() => onDismiss(toast.id)}>
            <Icon name="close" size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}
