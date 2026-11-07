import { MenuIcon, SquarePenIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

const MODES = [
  ["complete", "Completar", "El modelo continúa tu texto"],
  ["chat", "Chat", "Conversación por turnos"],
];

// "3_modern/wiki_32m" → group "3_modern", option "wiki_32m": the closed picker then shows the
// short, distinctive part of the name, which still fits on a 360 px phone.
function groupModels(models) {
  const groups = new Map();
  for (const model of models) {
    const slash = model.name.lastIndexOf("/");
    const group = slash === -1 ? "" : model.name.slice(0, slash);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push({ ...model, label: model.name.slice(slash + 1) + (model.stage === "base" ? " (base)" : "") });
  }
  // Newest phase first ("5_sft" before "4_pretrain"): the conversational models lead the list.
  return [...groups].sort(([a], [b]) => b.localeCompare(a, "en", { numeric: true }));
}

const modelDetails = (model) =>
  `${(model.params / 1e6).toFixed(1)}M parámetros · contexto ${model.context} · paso ${model.step}`;

// Top of the conversation: menu and new-chat buttons (phones), the model picker and the mode.
export default function Header({ models, model, modelsState, onModelChange, mode, onModeChange, onOpenSidebar, onNewChat }) {
  const placeholder = { loading: "Cargando modelos…", error: "Sin conexión", ready: "Sin modelos" }[modelsState];

  return (
    <header className="flex h-14 shrink-0 items-center gap-1 border-b px-2 md:gap-2 md:px-4">
      <Button variant="ghost" size="icon" className="md:hidden" aria-label="Abrir menú" onClick={onOpenSidebar}>
        <MenuIcon className="size-5" />
      </Button>

      <Select value={model?.name ?? ""} onValueChange={onModelChange} disabled={models.length === 0}>
        <SelectTrigger
          aria-label="Modelo"
          className="min-w-0 flex-1 border-0 bg-transparent font-semibold shadow-none hover:bg-accent md:max-w-xs md:flex-none dark:bg-transparent dark:hover:bg-accent [&_svg:not([class*='text-'])]:text-foreground"
        >
          {/* The closed picker shows only the short name; the list adds each model's details. */}
          <SelectValue placeholder={placeholder}>{model && groupModels([model])[0][1][0].label}</SelectValue>
        </SelectTrigger>
        <SelectContent position="popper" align="start" className="max-w-[calc(100vw-1rem)]">
          {groupModels(models).map(([group, items]) => (
            <SelectGroup key={group}>
              <SelectLabel>{group || "Modelos"}</SelectLabel>
              {items.map((m) => (
                <SelectItem key={m.name} value={m.name} textValue={m.label}>
                  <span className="flex flex-col items-start gap-0">
                    <span>{m.label}</span>
                    <span className="text-xs text-muted-foreground">{modelDetails(m)}</span>
                  </span>
                </SelectItem>
              ))}
            </SelectGroup>
          ))}
        </SelectContent>
      </Select>

      <ToggleGroup
        type="single"
        aria-label="Modo"
        spacing={0.5}
        className="ml-auto rounded-lg bg-muted p-0.5"
        value={mode}
        onValueChange={(value) => value && onModeChange(value)} // clicking the active mode keeps it
      >
        {/* A plain title, not a Radix Tooltip: the tooltip trigger would overwrite the item's
            data-state="on", which is what marks the active mode. */}
        {MODES.map(([value, text, description]) => (
          <ToggleGroupItem
            key={value}
            value={value}
            title={description}
            className="h-8 rounded-md px-2.5 text-muted-foreground hover:bg-transparent hover:text-foreground data-[state=on]:bg-background data-[state=on]:text-foreground data-[state=on]:shadow-xs sm:px-3"
          >
            {text}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>

      <Button variant="ghost" size="icon" className="md:hidden" aria-label="Nueva conversación" onClick={onNewChat}>
        <SquarePenIcon className="size-[19px]" />
      </Button>
    </header>
  );
}
