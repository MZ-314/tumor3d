import { useCallback, useRef, useState } from "react";
import type { ChatMessage } from "@shared/index";
import { reconstructFromFile } from "@viewer/api";
import { MessageList } from "./components/MessageList";
import { ComposeBar } from "./components/ComposeBar";

const WELCOME: ChatMessage = {
  id: "welcome",
  role: "assistant",
  text: "Upload any image and I'll generate a high-fidelity 3D model using SAM 2 + TRELLIS.2. Best results with a single clear object on a plain background.",
};

function uid(): string {
  return Math.random().toString(36).slice(2, 11);
}

export function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME]);
  const [busy, setBusy] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
    });
  }, []);

  const handleSend = useCallback(
    async (file: File, text?: string) => {
      const userMsg: ChatMessage = {
        id: uid(),
        role: "user",
        text: text || "Generate 3D model",
        attachmentUrl: URL.createObjectURL(file),
        attachmentName: file.name,
      };

      const loadingId = uid();
      const loadingMsg: ChatMessage = {
        id: loadingId,
        role: "assistant",
        loading: true,
        text: "Analyzing image and generating 3D model…",
      };

      setMessages((prev) => [...prev, userMsg, loadingMsg]);
      setBusy(true);
      scrollToBottom();

      try {
        const reconstruction = await reconstructFromFile(file);
        const assistantMsg: ChatMessage = {
          id: loadingId,
          role: "assistant",
          text: reconstruction.assistant_summary,
          reconstruction,
        };
        setMessages((prev) => prev.map((m) => (m.id === loadingId ? assistantMsg : m)));
      } catch (err) {
        const message = err instanceof Error ? err.message : "Reconstruction failed";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingId
              ? { ...m, loading: false, error: message, text: `I couldn't process that image. ${message}` }
              : m,
          ),
        );
      } finally {
        setBusy(false);
        scrollToBottom();
      }
    },
    [scrollToBottom],
  );

  return (
    <div className="app">
      <header className="app__header">
        <div>
          <h1>Image to 3D Assistant</h1>
          <p className="app__subtitle">SAM 2 + TRELLIS.2 + Blender pipeline</p>
        </div>
        <span className="app__badge">Prototype</span>
      </header>

      <main className="app__main" ref={listRef}>
        <MessageList messages={messages} />
      </main>

      <ComposeBar onSend={handleSend} disabled={busy} />
    </div>
  );
}
