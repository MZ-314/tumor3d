import type { ChatMessage } from "@shared/index";
import { API_BASE, resolveAssetUrl } from "@viewer/api";
import { MeshViewer } from "@viewer/MeshViewer";
import { ViewerErrorBoundary } from "@viewer/ViewerErrorBoundary";
import { AnalysisProgress } from "./AnalysisProgress";

interface MessageBubbleProps {
  message: ChatMessage;
}

function attachmentSrc(url: string): string {
  if (url.startsWith("blob:") || url.startsWith("http")) return url;
  return resolveAssetUrl(url, API_BASE);
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={`bubble-row bubble-row--${message.role}`}>
      <div
        className={`bubble bubble--${message.role}${message.reconstruction ? " bubble--has-viewer" : ""}`}
      >
        {!isUser && <span className="bubble__avatar">AI</span>}

        <div className="bubble__content">
          {message.text && <p className="bubble__text">{message.text}</p>}

          {message.loading && message.analysisStartedAt != null && message.analysisSliceCount != null ? (
            <AnalysisProgress
              sliceCount={message.analysisSliceCount}
              startedAt={message.analysisStartedAt}
              mode={message.analysisMode}
            />
          ) : message.loading ? (
            <div className="bubble__spinner" aria-label="Loading">
              <span />
              <span />
              <span />
            </div>
          ) : null}

          {message.attachmentUrl && (
            <figure className="bubble__attachment">
              <img src={attachmentSrc(message.attachmentUrl)} alt={message.attachmentName ?? "Scan"} />
              {message.attachmentName && (
                <figcaption>{message.attachmentName}</figcaption>
              )}
            </figure>
          )}

          {message.reconstruction && (
            <div className="bubble__viewer">
              <ViewerErrorBoundary>
                <MeshViewer reconstruction={message.reconstruction} apiBase={API_BASE} variant="chat" />
              </ViewerErrorBoundary>
            </div>
          )}

          {message.error && <p className="bubble__error">{message.error}</p>}
        </div>
      </div>
    </div>
  );
}
