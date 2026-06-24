import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatMessage, ChatSummary } from "@shared/index";
import {
  chatRecordToMessage,
  createChat,
  getChat,
  listChats,
  reconstructFromFiles,
} from "@viewer/api";
import { MessageList } from "./components/MessageList";
import { ComposeBar, type ScanModality } from "./components/ComposeBar";
import { ChatSidebar } from "./components/ChatSidebar";
import { StubBanner } from "./components/StubBanner";

const WELCOME: ChatMessage = {
  id: "welcome",
  role: "assistant",
  text:
    "**Core:** upload **one photo** → **AI 3D mesh** (TripoSR on RunPod). " +
    "**Also:** **DICOM volume** for real MRI/CT stacks (knee, brain, etc.) in the volume viewer. " +
    "**Optional:** **Brain + tumor AI** for MONAI segmentation. Pick the mode in the dropdown before sending.",
};

function uid(): string {
  return Math.random().toString(36).slice(2, 11);
}

export function App() {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME]);
  const [busy, setBusy] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const listRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
    });
  }, []);

  const refreshChats = useCallback(async () => {
    try {
      const list = await listChats();
      setChats(list);
    } catch {
      /* backend may be offline during dev */
    }
  }, []);

  useEffect(() => {
    void refreshChats();
  }, [refreshChats]);

  const loadChat = useCallback(
    async (chatId: string) => {
      setActiveChatId(chatId);
      try {
        const detail = await getChat(chatId);
        const msgs = detail.messages.map(chatRecordToMessage);
        setMessages(msgs.length ? msgs : [WELCOME]);
      } catch {
        setMessages([WELCOME]);
      }
      scrollToBottom();
    },
    [scrollToBottom],
  );

  const handleNewChat = useCallback(async () => {
    const chat = await createChat();
    setChats((prev) => [chat, ...prev]);
    setActiveChatId(chat.id);
    setMessages([WELCOME]);
  }, []);

  const handleSend = useCallback(
    async (files: File[], text?: string, modality: ScanModality = "ai_3d") => {
      setBusy(true);

      let chatId = activeChatId;
      try {
        if (!chatId) {
          const chat = await createChat();
          chatId = chat.id;
          setActiveChatId(chatId);
          setChats((prev) => [chat, ...prev]);
        }
      } catch (err) {
        setBusy(false);
        const message = err instanceof Error ? err.message : "Could not reach API";
        alert(
          `Cannot connect to the API. Check that RunPod is running and VITE_API_BASE in frontend/.env is correct.\n\n${message}`,
        );
        return;
      }

      const previewUrl = URL.createObjectURL(files[0]);
      const names = files.map((f) => f.name).join(", ");
      const userMsg: ChatMessage = {
        id: uid(),
        role: "user",
        text: text || `Uploaded ${files.length} slice(s)`,
        attachmentUrl: previewUrl,
        attachmentName: names,
      };

      const loadingId = uid();
      const analysisStartedAt = Date.now();
      const loadingMsg: ChatMessage = {
        id: loadingId,
        role: "assistant",
        loading: true,
        analysisSliceCount: files.length,
        analysisStartedAt,
        analysisMode: modality,
      };

      setMessages((prev) => [...prev.filter((m) => m.id !== "welcome"), userMsg, loadingMsg]);
      scrollToBottom();

      try {
        const reconstruction = await reconstructFromFiles(files, {
          chatId,
          text,
          modality,
        });
        const assistantMsg: ChatMessage = {
          id: loadingId,
          role: "assistant",
          text: reconstruction.assistant_summary,
          reconstruction,
        };
        setMessages((prev) => prev.map((m) => (m.id === loadingId ? assistantMsg : m)));
        void refreshChats();
      } catch (err) {
        const message = err instanceof Error ? err.message : "Analysis failed";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === loadingId
              ? { ...m, loading: false, error: message, text: `I couldn't process those slices. ${message}` }
              : m,
          ),
        );
      } finally {
        setBusy(false);
        scrollToBottom();
      }
    },
    [activeChatId, refreshChats, scrollToBottom],
  );

  return (
    <div className="app-layout">
      <ChatSidebar
        chats={chats}
        activeId={activeChatId}
        open={sidebarOpen}
        onToggle={() => setSidebarOpen((o) => !o)}
        onSelect={(id) => void loadChat(id)}
        onNew={() => void handleNewChat()}
      />

      <div className="app">
        <header className="app__header">
          <div>
            <h1>Meddollina Medical Chat</h1>
            <p className="app__subtitle">
              Photo → AI 3D mesh · DICOM stack → volume · optional brain tumor AI
            </p>
          </div>
          <span className="app__badge">Prototype</span>
        </header>

        <StubBanner />

        <main className="app__main" ref={listRef}>
          <MessageList messages={messages} />
        </main>

        <ComposeBar onSend={handleSend} disabled={busy} />
      </div>
    </div>
  );
}
