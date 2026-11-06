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
    <div className="welcome">
      <span className="welcome-mark" aria-hidden="true">L</span>
      <h2>{title}</h2>
      <p>{subtitle}</p>
      <ul className="suggestions" aria-label="Sugerencias">
        {suggestions.map((text) => (
          <li key={text}>
            <button type="button" disabled={disabled} onClick={() => onSuggestion(text)}>
              {text}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
