import { AlignLeftIcon, EllipsisIcon, MessageSquareIcon, PencilIcon, Trash2Icon } from "lucide-react";
import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

const MODE_LABEL = { complete: "Completar", chat: "Chat" };

// One row of the history: the title opens it; the ⋯ menu renames it in place or deletes it.
export default function ConversationItem({ conversation, active, onSelect, onRename, onAskDelete }) {
  const [editing, setEditing] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const titleRef = useRef(null);
  // false while a rename is open. Esc unmounts the focused field, which fires one last blur:
  // this flag makes that blur a no-op.
  const finished = useRef(true);

  function finishRename(input, save) {
    if (finished.current) return;
    finished.current = true;
    const clean = input.value.trim();
    setEditing(false);
    if (save && clean && clean !== conversation.title) onRename(conversation.id, clean);
    // Give the focus back to the row, where the keyboard user started.
    requestAnimationFrame(() => titleRef.current?.focus());
  }

  if (editing) {
    return (
      <li className="py-0.5">
        <Input
          className="h-9 bg-background"
          defaultValue={conversation.title}
          autoFocus
          aria-label="Nuevo nombre de la conversación"
          maxLength={120}
          onFocus={(e) => e.target.select()}
          onBlur={(e) => finishRename(e.target, true)}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.target.blur(); // blur saves, so Enter and clicking away agree
            if (e.key === "Escape") {
              e.stopPropagation(); // Esc cancels the rename only
              finishRename(e.target, false);
            }
          }}
        />
      </li>
    );
  }

  const ModeIcon = conversation.mode === "chat" ? MessageSquareIcon : AlignLeftIcon;
  return (
    <li
      className={cn(
        "group relative flex h-9 items-center rounded-lg transition-colors hover:bg-sidebar-accent max-md:h-10.5",
        (active || menuOpen) && "bg-sidebar-accent",
      )}
    >
      <button
        ref={titleRef}
        type="button"
        className={cn(
          "flex h-full min-w-0 flex-1 items-center gap-2.5 rounded-lg pr-9 pl-3 text-left text-sm text-secondary-text outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50",
          active && "font-medium text-foreground",
        )}
        title={conversation.title}
        aria-current={active ? "page" : undefined}
        onClick={() => onSelect(conversation.id)}
      >
        <ModeIcon className="size-[15px] shrink-0 opacity-70" aria-hidden="true" />
        <span className="sr-only">{MODE_LABEL[conversation.mode]}: </span>
        <span className="truncate">{conversation.title}</span>
      </button>
      <DropdownMenu onOpenChange={setMenuOpen}>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon-xs"
            className="absolute right-1 text-muted-foreground opacity-0 group-hover:opacity-100 hover:bg-background hover:text-foreground focus-visible:opacity-100 data-[state=open]:opacity-100 [@media(hover:none)]:opacity-100"
            aria-label={`Opciones de «${conversation.title}»`}
          >
            <EllipsisIcon />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="end"
          // The rename field takes the focus itself: the menu must not send it back to the trigger.
          onCloseAutoFocus={(e) => !finished.current && e.preventDefault()}
        >
          <DropdownMenuItem
            onSelect={() => {
              finished.current = false;
              setEditing(true);
            }}
          >
            <PencilIcon />
            Renombrar
          </DropdownMenuItem>
          <DropdownMenuItem variant="destructive" onSelect={() => onAskDelete(conversation)}>
            <Trash2Icon />
            Eliminar
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </li>
  );
}
