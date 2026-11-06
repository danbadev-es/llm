import { ArrowDownIcon } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import useAutoScroll from "../hooks/useAutoScroll.js";
import Message from "./Message.jsx";
import Welcome from "./Welcome.jsx";

function ThreadSkeleton() {
  return (
    <div className="flex flex-col gap-3.5 pt-2" aria-hidden="true">
      <Skeleton className="mb-4 h-10 w-2/5 self-end rounded-2xl" />
      <Skeleton className="h-3.5 w-[92%]" />
      <Skeleton className="h-3.5 w-[78%]" />
      <Skeleton className="h-3.5 w-[60%]" />
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
    <div className="relative min-h-0 flex-1">
      {/* scrollbar-gutter: the column does not shift sideways once the conversation scrolls. */}
      <div className="h-full overflow-y-auto overscroll-contain [scrollbar-gutter:stable_both-edges]" ref={scrollRef}>
        <div className="mx-auto max-w-[47rem] px-4 pt-4 pb-8 md:px-6 md:pt-6 md:pb-10" ref={contentRef} aria-busy={loading}>
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
                previousRole={messages[i - 1]?.role}
                isLast={i === last}
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
        <button
          type="button"
          onClick={jumpToLatest}
          className="absolute bottom-3 left-1/2 flex h-8.5 -translate-x-1/2 animate-fade-up items-center gap-2 rounded-full border bg-popover pr-4 pl-3 text-sm font-medium shadow-md transition-colors hover:bg-bubble"
        >
          <ArrowDownIcon className="size-4" />
          Ir al final
        </button>
      )}
    </div>
  );
}
