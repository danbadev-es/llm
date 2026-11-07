import { CalculatorIcon, ChevronDownIcon, ClockIcon, FileSearchIcon, LoaderIcon, TriangleAlertIcon, WrenchIcon } from "lucide-react";

const TOOLS = {
  calculadora: { label: "Calculadora", icon: CalculatorIcon, argument: (input) => input.expresion },
  fecha_hora: { label: "Fecha y hora", icon: ClockIcon, argument: () => "" },
  buscar_documentos: { label: "Tus documentos", icon: FileSearchIcon, argument: (input) => input.consulta && `«${input.consulta}»` },
};
const INVALID = { label: "Llamada no válida", icon: TriangleAlertIcon, argument: (input) => input.raw };

// One tool call inside an answer: the tool, its argument and, once it has run, the result.
// Short results (a number, a date) fit in the summary line; long ones (document passages) open below.
export default function ToolCall({ step, running }) {
  const tool = step.name ? (TOOLS[step.name] ?? { label: step.name, icon: WrenchIcon, argument: (i) => JSON.stringify(i) }) : INVALID;
  const Icon = tool.icon;
  const argument = tool.argument(step.input ?? {}) ?? "";
  const result = step.result ?? "";
  const short = result && result.length <= 48 && !result.includes("\n");
  const failed = result.startsWith("error:") || !step.name;

  return (
    <details className="group/tool my-2 rounded-xl border bg-sidebar text-sm leading-normal whitespace-normal">
      <summary className="flex cursor-pointer list-none items-center gap-2 rounded-xl px-3 py-2 outline-none select-none focus-visible:ring-[3px] focus-visible:ring-ring/50 [&::-webkit-details-marker]:hidden">
        <Icon className={failed ? "size-4 shrink-0 text-destructive" : "size-4 shrink-0 text-muted-foreground"} aria-hidden="true" />
        <span className="shrink-0 font-medium">{tool.label}</span>
        {argument && <span className="min-w-0 truncate font-mono text-[13px] text-secondary-text">{argument}</span>}
        {short && (
          <span className={failed ? "min-w-0 truncate text-destructive" : "min-w-0 truncate font-semibold"}>→ {result}</span>
        )}
        {running ? (
          <LoaderIcon className="ml-auto size-4 shrink-0 animate-spin text-muted-foreground" aria-label="Ejecutando" />
        ) : (
          <ChevronDownIcon
            className="ml-auto size-4 shrink-0 text-muted-foreground transition-transform group-open/tool:rotate-180"
            aria-hidden="true"
          />
        )}
      </summary>
      <div className="border-t px-3 py-2 text-[13px] whitespace-pre-wrap text-secondary-text">
        {running ? "Ejecutando…" : result || "Sin resultado."}
      </div>
    </details>
  );
}

// An answer's text with its tool calls placed where they happened (step.at: the text before them).
export function AnswerWithTools({ content, steps, streaming }) {
  const parts = [];
  let position = 0;
  steps.forEach((step, i) => {
    const at = Math.min(step.at ?? 0, content.length);
    if (at > position) parts.push(<span key={`text-${i}`}>{content.slice(position, at)}</span>);
    parts.push(<ToolCall key={step.id ?? i} step={step} running={streaming && step.result === undefined} />);
    position = Math.max(position, at);
  });
  if (position < content.length) parts.push(<span key="text-end">{content.slice(position)}</span>);
  return parts;
}
