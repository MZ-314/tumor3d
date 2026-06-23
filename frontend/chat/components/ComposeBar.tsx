import { useRef, useState, type DragEvent, type FormEvent } from "react";

interface ComposeBarProps {
  onSend: (file: File, text?: string) => void;
  disabled?: boolean;
}

export function ComposeBar({ onSend, disabled }: ComposeBarProps) {
  const [text, setText] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const submitFile = (file: File) => {
    if (!file.type.startsWith("image/") && !file.name.toLowerCase().endsWith(".dcm")) {
      alert("Please upload a PNG, JPEG, WebP, or image file.");
      return;
    }
    onSend(file, text.trim() || undefined);
    setText("");
    if (fileRef.current) fileRef.current.value = "";
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    const file = fileRef.current?.files?.[0];
    if (!file) {
      alert("Attach an image first.");
      return;
    }
    submitFile(file);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) submitFile(file);
  };

  return (
    <form
      className={`compose ${dragOver ? "compose--drag" : ""}`}
      onSubmit={onSubmit}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="compose__file"
        id="scan-file"
        disabled={disabled}
      />
      <label htmlFor="scan-file" className="compose__attach" title="Attach scan">
        📎
      </label>
      <input
        type="text"
        className="compose__input"
        placeholder="Upload an image… or drag & drop"
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={disabled}
      />
      <button type="submit" className="compose__send" disabled={disabled}>
        Send
      </button>
    </form>
  );
}
