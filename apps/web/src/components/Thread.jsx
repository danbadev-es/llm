import useAutoScroll from "../hooks/useAutoScroll.js";
import Icon from "./Icon.jsx";
import Message from "./Message.jsx";
import Welcome from "./Welcome.jsx";

function ThreadSkeleton() {
  return (
    <div className="thread-skeleton" aria-hidden="true">
      <span className="skeleton-line user" />
      <span className="skeleton-line" style={{ width: "92%" }} />
      <span className="skeleton-line" style={{ width: "78%" }} />
      <span className="skeleton-line" style={{ width: "60%" }} />
    </div>
  );
}

// The scrolling conversation. Only the last exchange can be regenerated or edited: the history
// is a straight line (no branches), so changing an older turn would orphan everything after it.
export default function Thread({
  mode,
  messages,
  loading,
  busy,
  canSend,
  viewKey,
  pinKey,
  notice,
  onSuggestion,
  onCopy,
  onRegenerate,
  onEdit,
}) {
  const { scrollRef, contentRef, atBottom, jumpToLatest } = useAutoScroll(pinKey);
  const last = messages.length - 1;
  const editable = mode === "chat" ? messages.findLastIndex((m) => m.role === "user") : last;

  return (
    <div className="thread">
      <div className="thread-scroll" ref={scrollRef}>
        <div className="thread-content" ref={contentRef} aria-busy={loading}>
          {notice}
          {loading ? (
            <ThreadSkeleton />
          ) : messages.length === 0 ? (
            <Welcome mode={mode} disabled={!canSend} onSuggestion={onSuggestion} />
          ) : (
            messages.map((message, i) => (
              <Message
                // viewKey changes with the conversation on screen: a half-open editor or a
                // "copied" check never leaks into the message at the same position of another one.
                key={`${viewKey}:${i}`}
                message={message}
                canEdit={!busy && canSend && i === editable}
                canRegenerate={!busy && canSend && i === last && message.role === "assistant"}
                onCopy={onCopy}
                onRegenerate={onRegenerate}
                onEdit={onEdit}
              />
            ))
          )}
        </div>
      </div>
      {!atBottom && !loading && messages.length > 0 && (
        <button type="button" className="jump-button" onClick={jumpToLatest}>
          <Icon name="arrowDown" size={16} />
          Ir al final
        </button>
      )}
    </div>
  );
}
