import type { ChatMessage } from "../types";
import { ExplainabilityCollapse } from "./ExplainabilityCollapse";
import { MarkdownMessage } from "./MarkdownMessage";

export function AnswerMessage({ message }: { message: ChatMessage }) {
  return (
    <article className="message assistant answer-message">
      <ExplainabilityCollapse message={message} />
      <div className="answer-content">
        <MarkdownMessage content={message.content} />
      </div>
      {message.meta ? <div className="mt-3 text-xs text-stone-500">{message.meta}</div> : null}
    </article>
  );
}
