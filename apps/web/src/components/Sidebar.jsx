import { SlidersHorizontalIcon, SquarePenIcon, XIcon } from "lucide-react";
import { useState } from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { groupByDate } from "../format.js";
import ConversationItem from "./ConversationItem.jsx";

export function BrandMark({ className = "" }) {
  return (
    <span
      className={`grid shrink-0 place-items-center rounded-lg bg-primary font-bold text-primary-foreground ${className}`}
      aria-hidden="true"
    >
      L
    </span>
  );
}

function HistorySkeleton() {
  return (
    <div className="flex flex-col gap-3 px-3 pt-2" aria-hidden="true">
      {[72, 55, 85, 62, 48].map((width, i) => (
        <Skeleton key={i} className="h-3.5" style={{ width: `${width}%` }} />
      ))}
    </div>
  );
}

// The history column's content, shared by the fixed column (desktop) and the drawer (phones).
function SidebarContent({ conversations, currentId, onSelect, onNewChat, onRename, onAskDelete, onOpenSettings, onClose }) {
  return (
    <div className="flex h-full flex-col bg-sidebar">
      <div className="flex h-14 items-center gap-2.5 px-4">
        <BrandMark className="size-7 text-sm" />
        <span className="text-lg font-bold tracking-tight">Leon</span>
        <span className="rounded-full border px-2 text-xs text-muted-foreground">local</span>
        {onClose && (
          <Button variant="ghost" size="icon-sm" className="ml-auto" aria-label="Cerrar menú" onClick={onClose}>
            <XIcon />
          </Button>
        )}
      </div>

      <div className="px-2">
        <Button
          variant="outline"
          className="h-10 w-full justify-start gap-2.5 rounded-lg bg-background px-3 text-[15px] font-semibold"
          title="Nueva conversación (Ctrl+Shift+O)"
          onClick={onNewChat}
        >
          <SquarePenIcon className="size-[17px]" />
          Nueva conversación
        </Button>
      </div>

      <nav
        className="mt-4 min-h-0 flex-1 overflow-y-auto px-2 pb-4"
        aria-label="Historial de conversaciones"
        aria-busy={conversations === null}
      >
        {conversations === null ? (
          <HistorySkeleton />
        ) : conversations.length === 0 ? (
          <p className="px-3 text-sm text-muted-foreground">Tus conversaciones aparecerán aquí.</p>
        ) : (
          groupByDate(conversations).map((group, i) => (
            <section key={group.label} aria-labelledby={`history-group-${i}`} className="not-first:mt-5">
              <h3 id={`history-group-${i}`} className="px-3 pb-1.5 text-xs font-semibold text-muted-foreground">
                {group.label}
              </h3>
              <ul>
                {group.items.map((conversation) => (
                  <ConversationItem
                    key={conversation.id}
                    conversation={conversation}
                    active={conversation.id === currentId}
                    onSelect={onSelect}
                    onRename={onRename}
                    onAskDelete={onAskDelete}
                  />
                ))}
              </ul>
            </section>
          ))
        )}
      </nav>

      <div className="border-t p-2">
        <Button variant="ghost" className="h-10 w-full justify-start gap-2.5 px-3 text-[15px]" onClick={onOpenSettings}>
          <SlidersHorizontalIcon className="size-[17px]" />
          Ajustes
        </Button>
      </div>
    </div>
  );
}

// The history: a fixed column on desktop, a drawer (Radix dialog: focus trap, Esc, overlay) on phones.
export default function Sidebar({ isPhone, open, onOpenChange, onDelete, ...props }) {
  const [toDelete, setToDelete] = useState(null);
  const content = <SidebarContent {...props} onAskDelete={setToDelete} />;

  return (
    <>
      {isPhone ? (
        <Sheet open={open} onOpenChange={onOpenChange}>
          <SheetContent side="left" showCloseButton={false} className="w-[min(300px,86vw)] gap-0 border-r-0">
            <SheetTitle className="sr-only">Menú</SheetTitle>
            <SheetDescription className="sr-only">Historial de conversaciones y ajustes</SheetDescription>
            <SidebarContent {...props} onAskDelete={setToDelete} onClose={() => onOpenChange(false)} />
          </SheetContent>
        </Sheet>
      ) : (
        <aside className="w-[268px] shrink-0 border-r" aria-label="Barra lateral">
          {content}
        </aside>
      )}

      <AlertDialog open={toDelete !== null} onOpenChange={(next) => !next && setToDelete(null)}>
        <AlertDialogContent className="rounded-xl bg-popover sm:max-w-md">
          <AlertDialogHeader>
            <AlertDialogTitle>¿Eliminar conversación?</AlertDialogTitle>
            <AlertDialogDescription>
              Se borrará <b className="font-semibold text-foreground">{toDelete?.title}</b> con todos sus mensajes. No se
              puede deshacer.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancelar</AlertDialogCancel>
            <AlertDialogAction variant="destructive" onClick={() => onDelete(toDelete.id)}>
              Eliminar
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
