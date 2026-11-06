import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

// How close to the bottom (px) still counts as "at the bottom".
const THRESHOLD = 48;
// Fallback end of a smooth jump, for browsers without the "scrollend" event.
const JUMP_TIMEOUT_MS = 900;

// Smart auto-scroll for a streaming thread, like ChatGPT or Claude:
// - while the reader is at the bottom, new text keeps the view pinned there;
// - scrolling up stops following, and `atBottom` turns false so a "jump to latest" button can show;
// - changing `pinKey` (a new message was sent, another conversation opened) pins to the bottom again.
// A ResizeObserver on the content, instead of an effect on every message update, catches every
// height change (streamed tokens, an error box appearing, fonts loading) exactly once per layout.
export default function useAutoScroll(pinKey) {
  const scrollRef = useRef(null);
  const contentRef = useRef(null);
  const following = useRef(true);
  // True during a smooth "jump to latest". Its intermediate positions are not the reader leaving,
  // and snapping while it animates would fight it: new text is caught up when it ends instead.
  const jumping = useRef(false);
  const [atBottom, setAtBottom] = useState(true);

  const measure = useCallback(() => {
    const box = scrollRef.current;
    const bottom = box.scrollHeight - box.scrollTop - box.clientHeight <= THRESHOLD;
    if (!jumping.current) following.current = bottom;
    setAtBottom(bottom);
  }, []);

  const endJump = useCallback(() => {
    if (!jumping.current) return;
    jumping.current = false;
    const box = scrollRef.current;
    if (following.current) box.scrollTop = box.scrollHeight; // text that arrived during the animation
    measure();
  }, [measure]);

  useEffect(() => {
    const box = scrollRef.current;
    // Wheel or touch upwards is an explicit wish to read back: stop following right away,
    // even before the scroll moves us past THRESHOLD.
    const leave = (e) => {
      if (e.type === "wheel" && e.deltaY >= 0) return;
      jumping.current = false;
      following.current = false;
    };
    const observer = new ResizeObserver(() => {
      if (following.current && !jumping.current) box.scrollTop = box.scrollHeight;
      measure();
    });
    observer.observe(contentRef.current);
    box.addEventListener("scroll", measure, { passive: true });
    box.addEventListener("scrollend", endJump);
    box.addEventListener("wheel", leave, { passive: true });
    box.addEventListener("touchmove", leave, { passive: true });
    return () => {
      observer.disconnect();
      box.removeEventListener("scroll", measure);
      box.removeEventListener("scrollend", endJump);
      box.removeEventListener("wheel", leave);
      box.removeEventListener("touchmove", leave);
    };
  }, [measure, endJump]);

  // Layout effect: pin before the browser paints, so a new conversation never flashes at the top.
  useLayoutEffect(() => {
    following.current = true;
    jumping.current = false;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [pinKey]);

  const jumpToLatest = useCallback(() => {
    const box = scrollRef.current;
    following.current = true;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      box.scrollTop = box.scrollHeight;
      return;
    }
    jumping.current = true;
    box.scrollTo({ top: box.scrollHeight, behavior: "smooth" });
    setTimeout(endJump, JUMP_TIMEOUT_MS);
  }, [endJump]);

  return { scrollRef, contentRef, atBottom, jumpToLatest };
}
