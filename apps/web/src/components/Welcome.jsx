import { BrandMark } from "./Sidebar.jsx";

// The empty state of a new conversation. The suggestions stay in Spanish on purpose:
// they are prompts for the model, which only knows Spanish.
const CONTENT = {
  complete: {
    title: "Empieza un texto",
    subtitle: "Escribe el comienzo y Leon lo continuará, token a token. Tu texto aparecerá resaltado.",
    suggestions: ["En un lugar de la Mancha", "La ciudad de Madrid es", "El sistema solar está formado por", "La Segunda Guerra Mundial"],
  },
  chat: {
    title: "¿En qué puedo ayudarte?",
    subtitle: "Conversa con Leon, un modelo de lenguaje entrenado desde cero.",
    suggestions: ["Hola, ¿quién eres?", "¿Cuál es la capital de Francia?", "Explícame qué es la fotosíntesis", "Cuéntame un chiste"],
  },
};

export default function Welcome({ mode, disabled, onSuggestion }) {
  const { title, subtitle, suggestions } = CONTENT[mode];
  return (
    <div className="flex animate-fade-up flex-col items-center pt-6 text-center md:pt-[12vh]">
      <BrandMark className="size-11 rounded-xl text-xl" />
      <h2 className="mt-5 text-[22px] font-semibold tracking-tight md:text-[28px]">{title}</h2>
      <p className="mt-2 max-w-md text-muted-foreground">{subtitle}</p>
      <ul className="mt-6 grid w-full max-w-xl grid-cols-1 gap-2 min-[421px]:grid-cols-2 md:mt-10" aria-label="Sugerencias">
        {suggestions.map((text) => (
          <li key={text}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onSuggestion(text)}
              className="min-h-11 w-full rounded-xl border bg-background px-4 py-2.5 text-left text-sm text-secondary-text transition-colors outline-none hover:border-input hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {text}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
