import { useRef, useState, type DragEvent, type FormEvent } from "react";

interface ComposeBarProps {
  onSend: (files: File[], text?: string) => void;
  disabled?: boolean;
}

const ACCEPT = "image/*,.dcm,.dicom,.jpg,.jpeg,.png,.webp";

const IMAGE_EXTENSIONS = new Set([
  ".jpg",
  ".jpeg",
  ".png",
  ".webp",
  ".gif",
  ".bmp",
  ".tif",
  ".tiff",
]);

function isValidFile(file: File): boolean {
  if (file.type.startsWith("image/")) return true;
  const n = file.name.toLowerCase();
  if (n.endsWith(".dcm") || n.endsWith(".dicom")) return true;
  const dot = n.lastIndexOf(".");
  if (dot === -1) return false;
  return IMAGE_EXTENSIONS.has(n.slice(dot));
}

export function ComposeBar({ onSend, disabled }: ComposeBarProps) {
  const [text, setText] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [picked, setPicked] = useState<File[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);

  const submitFiles = (files: File[]) => {
    const valid = files.filter(isValidFile);
    if (!valid.length) {
      alert("Please upload PNG, JPEG, WebP, or DICOM slices.");
      return;
    }
    onSend(valid, text.trim() || undefined);
    setText("");
    setPicked([]);
    if (fileRef.current) fileRef.current.value = "";
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    const fromInput = fileRef.current?.files ? Array.from(fileRef.current.files) : [];
    const files = picked.length ? picked : fromInput;
    if (!files.length) {
      alert("Attach at least one slice.");
      return;
    }
    submitFiles(files);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length) {
      setPicked(files);
      submitFiles(files);
    }
  };

  const onFileChange = () => {
    const files = fileRef.current?.files ? Array.from(fileRef.current.files) : [];
    setPicked(files);
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
        accept={ACCEPT}
        multiple
        className="compose__file"
        id="scan-file"
        disabled={disabled}
        onChange={onFileChange}
      />
      <label htmlFor="scan-file" className="compose__attach" title="Attach slice(s)">
        📎
      </label>
      <input
        type="text"
        className="compose__input"
        placeholder="Upload 1–N axial slices… drag & drop for better Z"
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={disabled}
      />
      {picked.length > 0 && (
        <span className="compose__count">{picked.length} file(s)</span>
      )}
      <button
        type="submit"
        className="compose__send"
        disabled={disabled || picked.length === 0}
        title={picked.length === 0 ? "Attach a slice first" : disabled ? "Processing…" : "Send"}
      >
        {disabled ? "Processing…" : "Send"}
      </button>
    </form>
  );
}
