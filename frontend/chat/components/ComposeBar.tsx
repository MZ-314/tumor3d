import { useRef, useState, type DragEvent, type FormEvent } from "react";

export type ScanModality = "ai_3d" | "volume_mri" | "brain_mri";

interface ComposeBarProps {
  onSend: (files: File[], text?: string, modality?: ScanModality) => void;
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

function isDicomFile(file: File): boolean {
  const n = file.name.toLowerCase();
  return n.endsWith(".dcm") || n.endsWith(".dicom");
}

function filterDicom(files: File[]): File[] {
  return files.filter(isDicomFile);
}

export function ComposeBar({ onSend, disabled }: ComposeBarProps) {
  const [text, setText] = useState("");
  const [modality, setModality] = useState<ScanModality>("ai_3d");
  const [dragOver, setDragOver] = useState(false);
  const [picked, setPicked] = useState<File[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);

  const submitFiles = (files: File[]) => {
    const valid = files.filter(isValidFile);
    if (!valid.length) {
      alert("Please upload PNG, JPEG, WebP, or DICOM slices.");
      return;
    }
    if (modality === "ai_3d" && valid.length !== 1) {
      alert("AI 3D mode uses one image. Switch to DICOM volume for multiple slices.");
      return;
    }
    if (modality !== "ai_3d" && valid.length === 1 && !isDicomFile(valid[0])) {
      const ok = window.confirm(
        "Single photo detected. Use AI 3D mode for one image, or OK to continue as a 1-slice volume.",
      );
      if (!ok) return;
    }
    onSend(valid, text.trim() || undefined, modality);
    setText("");
    setPicked([]);
    if (fileRef.current) fileRef.current.value = "";
    if (folderRef.current) folderRef.current.value = "";
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    const fromInput = fileRef.current?.files ? Array.from(fileRef.current.files) : [];
    const files = picked.length ? picked : fromInput;
    if (!files.length) {
      alert("Attach an image or DICOM series.");
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

  const onFolderChange = () => {
    const all = folderRef.current?.files ? Array.from(folderRef.current.files) : [];
    const dicoms = filterDicom(all);
    if (!dicoms.length) {
      alert("No .dcm files found in that folder.");
      return;
    }
    setModality("volume_mri");
    setPicked(dicoms);
    submitFiles(dicoms);
  };

  const showFolder = modality !== "ai_3d";

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
        multiple={modality !== "ai_3d"}
        className="compose__file"
        id="scan-file"
        disabled={disabled}
        onChange={onFileChange}
      />
      <input
        ref={folderRef}
        type="file"
        multiple
        className="compose__file"
        id="scan-folder"
        disabled={disabled}
        onChange={onFolderChange}
        {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
      />
      <label htmlFor="scan-file" className="compose__attach" title="Attach image or slices">
        📎
      </label>
      {showFolder && (
        <button
          type="button"
          className="compose__folder"
          disabled={disabled}
          title="Upload entire DICOM folder"
          onClick={() => folderRef.current?.click()}
        >
          📁
        </button>
      )}
      <select
        className="compose__modality"
        value={modality}
        disabled={disabled}
        onChange={(e) => setModality(e.target.value as ScanModality)}
        title="Reconstruction mode"
        aria-label="Reconstruction mode"
      >
        <option value="ai_3d">AI 3D (1 photo)</option>
        <option value="volume_mri">DICOM volume</option>
        <option value="brain_mri">Brain + tumor AI</option>
      </select>
      <input
        type="text"
        className="compose__input"
        placeholder={
          modality === "ai_3d"
            ? "One everyday photo only — not MRI montages or scan sheets…"
            : modality === "volume_mri"
              ? "DICOM series: 📁 folder or many .dcm files…"
              : "Brain DICOM + MONAI tumor segmentation…"
        }
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
        title={picked.length === 0 ? "Attach files first" : disabled ? "Processing…" : "Send"}
      >
        {disabled ? "Processing…" : "Send"}
      </button>
    </form>
  );
}
