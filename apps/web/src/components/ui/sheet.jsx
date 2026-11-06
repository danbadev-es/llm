import { Dialog as SheetPrimitive } from "radix-ui";
import { OVERLAY } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

const Sheet = (props) => <SheetPrimitive.Root data-slot="sheet" {...props} />;
const SheetTitle = ({ className, ...props }) => <SheetPrimitive.Title className={cn("font-semibold", className)} {...props} />;
const SheetDescription = ({ className, ...props }) => (
  <SheetPrimitive.Description className={cn("text-sm text-muted-foreground", className)} {...props} />
);

// A side panel (the history drawer on phones). It is a Radix dialog: focus stays inside,
// Esc and the overlay close it, and the page behind is hidden from screen readers.
function SheetContent({ className, children, side = "left", ...props }) {
  return (
    <SheetPrimitive.Portal>
      <SheetPrimitive.Overlay className={OVERLAY} />
      <SheetPrimitive.Content
        data-slot="sheet-content"
        className={cn(
          "fixed inset-y-0 z-50 flex h-full flex-col shadow-lg outline-none ease-in-out data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:duration-200 data-[state=open]:duration-300",
          side === "left"
            ? "left-0 border-r data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left"
            : "right-0 border-l data-[state=closed]:slide-out-to-right data-[state=open]:slide-in-from-right",
          className,
        )}
        {...props}
      >
        {children}
      </SheetPrimitive.Content>
    </SheetPrimitive.Portal>
  );
}

export { Sheet, SheetContent, SheetDescription, SheetTitle };
