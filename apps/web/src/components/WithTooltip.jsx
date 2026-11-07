import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

// An icon button's tooltip in one line: <WithTooltip label="Copiar"><Button …/></WithTooltip>.
// Not for Radix items that use data-state themselves (toggle items, menu triggers): the tooltip
// trigger would overwrite it.
export default function WithTooltip({ label, side = "top", children }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent side={side} sideOffset={4}>
        {label}
      </TooltipContent>
    </Tooltip>
  );
}
