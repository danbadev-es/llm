import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import Icon from "./Icon.jsx";

const GAP = 4; // px between the trigger and the menu
const MARGIN = 8; // px the menu keeps from the viewport edges

// A menu button following the WAI-ARIA "menu button" pattern:
// - the trigger opens it with a click, Enter, Space or ↓ (first item) / ↑ (last item);
// - ↑ ↓ Home End move between items, Enter activates, Esc closes and returns focus, Tab closes.
// The menu is rendered in a portal with fixed coordinates: inside the history list it would be
// clipped by the list's overflow, and it flips upwards when there is no room below the trigger.
// items: [{ label, icon, onSelect, danger?, keepFocus? }]. keepFocus: the action moves focus itself
// (e.g. to a rename field), so the menu must not send it back to the trigger.
export default function Menu({ label, items, className = "", onOpenChange }) {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef(null);
  const menuRef = useRef(null);
  const initialItem = useRef(0);

  function toggle(next, focusIndex = 0) {
    initialItem.current = focusIndex;
    setOpen(next);
    onOpenChange?.(next);
  }

  function close(returnFocus) {
    toggle(false);
    if (returnFocus) triggerRef.current?.focus();
  }

  // Place the menu before the browser paints it, then focus the requested item.
  useLayoutEffect(() => {
    if (!open) return;
    const menu = menuRef.current;
    const anchor = triggerRef.current.getBoundingClientRect();
    const { offsetWidth: width, offsetHeight: height } = menu;
    const below = anchor.bottom + GAP + height + MARGIN <= window.innerHeight;
    menu.style.top = `${below ? anchor.bottom + GAP : Math.max(MARGIN, anchor.top - GAP - height)}px`;
    menu.style.left = `${Math.min(Math.max(MARGIN, anchor.right - width), window.innerWidth - width - MARGIN)}px`;
    const buttons = menu.querySelectorAll('[role="menuitem"]');
    buttons[(initialItem.current + buttons.length) % buttons.length]?.focus();
  }, [open]);

  // Any pointer down outside, a scroll or a resize closes it (fixed coordinates would go stale).
  useEffect(() => {
    if (!open) return;
    const outside = (e) => {
      if (!menuRef.current?.contains(e.target) && !triggerRef.current?.contains(e.target)) close(false);
    };
    const dismiss = (e) => !menuRef.current?.contains(e.target) && close(false);
    document.addEventListener("pointerdown", outside);
    window.addEventListener("scroll", dismiss, true);
    window.addEventListener("resize", dismiss);
    return () => {
      document.removeEventListener("pointerdown", outside);
      window.removeEventListener("scroll", dismiss, true);
      window.removeEventListener("resize", dismiss);
    };
  });

  function onMenuKeyDown(e) {
    const buttons = [...menuRef.current.querySelectorAll('[role="menuitem"]')];
    const index = buttons.indexOf(document.activeElement);
    const moves = { ArrowDown: index + 1, ArrowUp: index - 1, Home: 0, End: buttons.length - 1 };
    if (e.key in moves) {
      e.preventDefault();
      buttons[(moves[e.key] + buttons.length) % buttons.length].focus();
    } else if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation(); // Esc closes the menu only, not the drawer or the generation behind it
      close(true);
    } else if (e.key === "Tab") {
      close(false);
    }
  }

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className={`icon-button ${className}`}
        aria-label={label}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => (open ? close(false) : toggle(true))}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            e.preventDefault();
            toggle(true, e.key === "ArrowDown" ? 0 : -1);
          }
        }}
      >
        <Icon name="more" />
      </button>
      {open &&
        createPortal(
          <div ref={menuRef} className="menu" role="menu" aria-label={label} onKeyDown={onMenuKeyDown}>
            {items.map((item) => (
              <button
                key={item.label}
                type="button"
                role="menuitem"
                tabIndex={-1}
                className={item.danger ? "danger" : ""}
                onClick={() => {
                  close(!item.keepFocus);
                  item.onSelect();
                }}
              >
                <Icon name={item.icon} size={16} />
                {item.label}
              </button>
            ))}
          </div>,
          document.body,
        )}
    </>
  );
}
