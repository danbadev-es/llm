import { AlertDialog as AlertDialogPrimitive } from "radix-ui";
import { buttonVariants } from "@/components/ui/button";
import { DIALOG_PANEL, OVERLAY } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

const AlertDialog = (props) => <AlertDialogPrimitive.Root data-slot="alert-dialog" {...props} />;

function AlertDialogContent({ className, ...props }) {
  return (
    <AlertDialogPrimitive.Portal>
      <AlertDialogPrimitive.Overlay className={OVERLAY} />
      <AlertDialogPrimitive.Content data-slot="alert-dialog-content" className={cn(DIALOG_PANEL, "sm:max-w-md", className)} {...props} />
    </AlertDialogPrimitive.Portal>
  );
}

const AlertDialogHeader = ({ className, ...props }) => <div className={cn("flex flex-col gap-2 text-left", className)} {...props} />;
const AlertDialogFooter = ({ className, ...props }) => (
  <div className={cn("flex flex-col-reverse gap-2 sm:flex-row sm:justify-end", className)} {...props} />
);
const AlertDialogTitle = ({ className, ...props }) => (
  <AlertDialogPrimitive.Title className={cn("text-lg font-semibold", className)} {...props} />
);
const AlertDialogDescription = ({ className, ...props }) => (
  <AlertDialogPrimitive.Description className={cn("text-sm leading-relaxed text-muted-foreground", className)} {...props} />
);
const AlertDialogAction = ({ className, variant = "default", ...props }) => (
  <AlertDialogPrimitive.Action className={cn(buttonVariants({ variant }), className)} {...props} />
);
const AlertDialogCancel = ({ className, ...props }) => (
  <AlertDialogPrimitive.Cancel className={cn(buttonVariants({ variant: "outline" }), className)} {...props} />
);

export {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
};
