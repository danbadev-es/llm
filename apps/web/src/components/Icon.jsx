// Inline SVG icons (shapes after the open-source Lucide set), so the app needs no icon dependency
// and every icon follows the text color. Decorative by default: the button around it carries the label.
const PATHS = {
  menu: ["M4 7h16", "M4 12h16", "M4 17h16"],
  close: ["M18 6 6 18", "m6 6 12 12"],
  more: [],
  pencil: [
    "M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z",
    "m15 5 4 4",
  ],
  trash: ["M3 6h18", "M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6", "M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"],
  sliders: ["M21 4h-7", "M10 4H3", "M21 12h-9", "M8 12H3", "M21 20h-5", "M12 20H3", "M14 2v4", "M8 10v4", "M16 18v4"],
  copy: ["M8 10a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H10a2 2 0 0 1-2-2z", "M4 16a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2"],
  check: ["M20 6 9 17l-5-5"],
  retry: ["M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8", "M3 3v5h5"],
  arrowUp: ["m5 12 7-7 7 7", "M12 19V5"],
  arrowDown: ["M12 5v14", "m19 12-7 7-7-7"],
  alert: ["M12 8v4", "M12 16h.01", "M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0"],
  chevronDown: ["m6 9 6 6 6-6"],
  newChat: [
    "M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7",
    "M18.375 2.625a1 1 0 0 1 3 3l-9.013 9.014a2 2 0 0 1-.853.505l-2.873.84a.5.5 0 0 1-.62-.62l.84-2.873a2 2 0 0 1 .506-.852z",
  ],
  text: ["M17 6H3", "M21 12H3", "M15 18H3"],
  chat: ["M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"],
};

export default function Icon({ name, size = 18, className = "" }) {
  return (
    <svg
      className={`icon ${className}`}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {name === "more" && [5, 12, 19].map((x) => <circle key={x} cx={x} cy="12" r="1.25" fill="currentColor" />)}
      {name === "stop" && <rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor" stroke="none" />}
      {PATHS[name]?.map((d) => <path key={d} d={d} />)}
    </svg>
  );
}
