import { Slider as SliderPrimitive } from "radix-ui";
import { cn } from "@/lib/utils";

// A single-thumb slider. id and the aria-* props go to the thumb, the element that has the
// focus and the slider role, so <Label htmlFor> and aria-describedby work as on an <input>.
function Slider({ className, id, value, min = 0, max = 100, "aria-describedby": describedBy, ...props }) {
  return (
    <SliderPrimitive.Root
      data-slot="slider"
      value={[value]}
      min={min}
      max={max}
      className={cn("relative flex w-full touch-none items-center py-2 select-none data-[disabled]:opacity-50", className)}
      {...props}
    >
      <SliderPrimitive.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-muted">
        <SliderPrimitive.Range className="absolute h-full bg-primary" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb
        id={id}
        aria-describedby={describedBy}
        className="block size-4 shrink-0 rounded-full border border-primary bg-background shadow-sm transition-[box-shadow] hover:ring-4 hover:ring-ring/25 focus-visible:ring-4 focus-visible:ring-ring/40 focus-visible:outline-hidden"
      />
    </SliderPrimitive.Root>
  );
}

export { Slider };
