import type { ChatMessage } from "@shared/index";
import { MeshViewer } from "@viewer/MeshViewer";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={`bubble-row bubble-row--${message.role}`}>
      <div className={`bubble bubble--${message.role}`}>
        {!isUser && <span className="bubble__avatar">AI</span>}

        <div className="bubble__content">
          {message.text && <p className="bubble__text">{message.text}</p>}

          {message.loading && (
            <div className="bubble__spinner" aria-label="Loading">
              <span />
              <span />
              <span />
            </div>
          )}

          {message.attachmentUrl && (
            <figure className="bubble__attachment">
              <img src={message.attachmentUrl} alt={message.attachmentName ?? "Scan"} />
              {message.attachmentName && (
                <figcaption>{message.attachmentName}</figcaption>
              )}
            </figure>
          )}

          {message.reconstruction && (
            <div className="bubble__viewer">
              <MeshViewer reconstruction={message.reconstruction} compact />
            </div>
          )}

          {message.error && <p className="bubble__error">{message.error}</p>}
        </div>
      </div>
    </div>
  );
}
