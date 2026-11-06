import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Kbd } from "@/components/ui/kbd";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { DEFAULT_SETTINGS } from "../preferences.js";

const SHORTCUTS = [
  [["Ctrl", "Shift", "O"], "Nueva conversación"],
  [["/"], "Escribir en el cuadro de texto"],
  [["Enter"], "Enviar"],
  [["Shift", "Enter"], "Nueva línea"],
  [["Esc"], "Detener la respuesta o cerrar"],
];

const Hint = ({ id, children }) => (
  <p id={id} className="text-[13px] leading-snug text-muted-foreground">
    {children}
  </p>
);

function SliderField({ id, label, hint, value, display, min, max, step, onChange }) {
  return (
    <div className="grid gap-1.5">
      <div className="flex items-center justify-between">
        <Label htmlFor={id}>{label}</Label>
        <output htmlFor={id} className="text-sm text-muted-foreground tabular-nums">
          {display}
        </output>
      </div>
      <Slider id={id} value={value} min={min} max={max} step={step} aria-describedby={`${id}-hint`} onValueChange={([v]) => onChange(v)} />
      <Hint id={`${id}-hint`}>{hint}</Hint>
    </div>
  );
}

// Changes apply (and persist) immediately: "Listo" only closes the dialog.
export default function SettingsDialog({ open, onClose, settings, onChange, maxTokensLimit }) {
  const set = (key) => (value) => onChange({ ...settings, [key]: value });
  const maxTokens = Math.min(settings.maxTokens, maxTokensLimit);

  return (
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="gap-6 sm:max-w-[30rem]">
        <DialogHeader>
          <DialogTitle>Ajustes</DialogTitle>
          <DialogDescription className="sr-only">Generación, instrucciones del sistema y atajos de teclado</DialogDescription>
        </DialogHeader>

        <div className="grid gap-6">
          <div className="grid gap-1.5">
            <Label htmlFor="system">Instrucciones del sistema</Label>
            <Textarea
              id="system"
              rows={3}
              className="resize-y"
              value={settings.system}
              placeholder="Eres Leon, un asistente que responde en español de forma breve."
              aria-describedby="system-hint"
              onChange={(e) => set("system")(e.target.value)}
            />
            <Hint id="system-hint">Solo en modo Chat: se envían al principio de cada conversación.</Hint>
          </div>
          <SliderField
            id="temperature"
            label="Temperatura"
            hint="Más baja: respuestas más seguras y repetitivas. Más alta: más variadas y arriesgadas."
            value={settings.temperature}
            min={0}
            max={1.5}
            step={0.05}
            display={settings.temperature === 0 ? "0 (greedy)" : settings.temperature.toFixed(2)}
            onChange={set("temperature")}
          />
          <SliderField
            id="top-k"
            label="Top-k"
            hint="Solo se elige entre los k tokens más probables. 0 = sin límite."
            value={settings.topK}
            min={0}
            max={200}
            step={5}
            display={settings.topK === 0 ? "sin límite" : settings.topK}
            onChange={set("topK")}
          />
          <SliderField
            id="max-tokens"
            label="Tokens máximos"
            hint={`Longitud máxima de cada respuesta (hasta ${maxTokensLimit} con este modelo).`}
            value={maxTokens}
            min={8}
            max={maxTokensLimit}
            step={8}
            display={maxTokens}
            onChange={set("maxTokens")}
          />

          <section aria-labelledby="shortcuts-title" className="border-t pt-5">
            <h3 id="shortcuts-title" className="mb-3 text-sm font-semibold">
              Atajos de teclado
            </h3>
            <dl className="grid gap-2 text-sm">
              {SHORTCUTS.map(([keys, action]) => (
                <div key={action} className="flex items-center justify-between gap-4">
                  <dt className="text-secondary-text">{action}</dt>
                  <dd className="flex gap-1">
                    {keys.map((key) => (
                      <Kbd key={key}>{key}</Kbd>
                    ))}
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onChange(DEFAULT_SETTINGS)}>
            Restablecer
          </Button>
          <Button autoFocus onClick={onClose}>
            Listo
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
