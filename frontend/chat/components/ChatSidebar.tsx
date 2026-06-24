import type { ChatSummary } from "@shared/index";

interface ChatSidebarProps {
  chats: ChatSummary[];
  activeId: string | null;
  open: boolean;
  onToggle: () => void;
  onSelect: (id: string) => void;
  onNew: () => void;
}

export function ChatSidebar({
  chats,
  activeId,
  open,
  onToggle,
  onSelect,
  onNew,
}: ChatSidebarProps) {
  return (
    <aside className={`sidebar ${open ? "sidebar--open" : "sidebar--collapsed"}`}>
      <div className="sidebar__head">
        <button type="button" className="sidebar__toggle" onClick={onToggle} aria-label="Toggle sidebar">
          {open ? "◀" : "▶"}
        </button>
        {open && (
          <button type="button" className="sidebar__new" onClick={onNew}>
            + New chat
          </button>
        )}
      </div>
      {open && (
        <ul className="sidebar__list">
          {chats.length === 0 && <li className="sidebar__empty">No saved chats yet</li>}
          {chats.map((chat) => (
            <li key={chat.id}>
              <button
                type="button"
                className={`sidebar__item ${chat.id === activeId ? "sidebar__item--active" : ""}`}
                onClick={() => onSelect(chat.id)}
              >
                <span className="sidebar__title">{chat.title}</span>
                <span className="sidebar__date">
                  {new Date(chat.updated_at).toLocaleDateString()}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
