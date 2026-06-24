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
import { ComposeBar } from "./components/ComposeBar";
import { ChatSidebar } from "./components/ChatSidebar";

const WELCOME: ChatMessage = {
  id: "welcome",
  role: "assistant",
  text:
    "Upload one or more brain MRI/CT slices. I'll estimate tumor location, show a rotatable 3D model, and report coordinates. More slices improve depth (Z). Research prototype — not for diagnosis.",
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
    async (files: File[], text?: string) => {
      let chatId = activeChatId;
      if (!chatId) {
        const chat = await createChat();
        chatId = chat.id;
        setActiveChatId(chatId);
        setChats((prev) => [chat, ...prev]);
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
      const loadingMsg: ChatMessage = {
        id: loadingId,
        role: "assistant",
        loading: true,
        text: `Segmenting ${files.length} slice(s) and building lesion meshes…`,
      };

      setMessages((prev) => [...prev.filter((m) => m.id !== "welcome"), userMsg, loadingMsg]);
      setBusy(true);
      scrollToBottom();

      try {
        const reconstruction = await reconstructFromFiles(files, {
          chatId,
          text,
          modality: "brain_mri",
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
              Tumor localization · 1–N slices · stub on CPU, MONAI on GPU
            </p>
          </div>
          <span className="app__badge">Prototype</span>
        </header>

        <main className="app__main" ref={listRef}>
          <MessageList messages={messages} />
        </main>

        <ComposeBar onSend={handleSend} disabled={busy} />
      </div>
    </div>
  );
}
