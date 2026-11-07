import { WrenchIcon } from "lucide-react";
import { Toggle } from "@/components/ui/toggle";

// Leon's tools (phase 6) on or off, next to the context meter. A title instead of a tooltip:
// the tooltip trigger would overwrite the toggle's own data-state.
export default function ToolsToggle({ on, onChange }) {
  return (
    <Toggle
      pressed={on}
      onPressedChange={onChange}
      aria-label="Herramientas"
      title={
        on
          ? "Herramientas activadas: calculadora, fecha y hora, y búsqueda en tus documentos (data/docs)"
          : "Herramientas desactivadas"
      }
      className="mb-[3px] size-8 min-w-8 shrink-0 rounded-full p-0 text-muted-foreground hover:text-foreground data-[state=on]:bg-accent data-[state=on]:text-primary"
    >
      <WrenchIcon className="size-[17px]" />
    </Toggle>
  );
}
