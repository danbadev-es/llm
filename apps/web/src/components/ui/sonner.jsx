import { Toaster as Sonner } from "sonner";

// shadcn's toaster with the app's colors. Its theme follows the system, like the rest of the app.
function Toaster(props) {
  return (
    <Sonner
      theme="system"
      className="toaster group"
      style={{
        "--normal-bg": "var(--popover)",
        "--normal-text": "var(--popover-foreground)",
        "--normal-border": "var(--border)",
        "--border-radius": "var(--radius)",
      }}
      {...props}
    />
  );
}

export { Toaster };
