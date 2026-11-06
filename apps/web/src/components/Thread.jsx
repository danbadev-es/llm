import { useEffect, useRef } from "react";

const SUGGESTIONS = {
  complete: ["En un lugar de la Mancha", "La ciudad de Madrid es", "El sistema solar está formado por", "La Segunda Guerra Mundial"],
  chat: ["Hola, ¿quién eres?", "¿Cuál es la capital de Francia?", "Explícame qué es la fotosíntesis", "Cuéntame un chiste"],
};

function Welcome({ mode, onSuggestion }) {
  return (
    <div className="welcome">
      <h2>{mode === "chat" ? "¿En qué puedo ayudarte?" : "Empieza un texto"}</h2>
      <p>{mode === "chat" ? "Conversa con Leon" : "Leon lo continuará token a token"}</p>
      <div className="suggestions">
        {SUGGESTIONS[mode].map((text) => (
          <button key={text} onClick={() => onSuggestion(text)}>{text}</button>
        ))}
      </div>
    </div>
  );
}

function Message({ message }) {
  return (
    <div className={`msg ${message.role}`}>
      <div>
        <div className="bubble">
          {message.prompt && <span className="prompt">{message.prompt}</span>}
          <span className={message.streaming ? "cursor" : ""}>{message.content}</span>
          {message.error && <span className="error">Error: {message.error}</span>}
        </div>
        {message.meta && <div className="meta">{message.meta}</div>}
      </div>
    </div>
  );
}

export default function Thread({ mode, messages, onSuggestion }) {
  const scrollRef = useRef(null);
  useEffect(() => {
    const box = scrollRef.current;
    box.scrollTop = box.scrollHeight; // follow the text as it streams in
  }, [messages]);

  return (
    <div className="messages" ref={scrollRef}>
      <div className="column">
        {messages.length === 0 ? (
          <Welcome mode={mode} onSuggestion={onSuggestion} />
        ) : (
          messages.map((message, i) => <Message key={i} message={message} />)
        )}
      </div>
    </div>
  );
}
