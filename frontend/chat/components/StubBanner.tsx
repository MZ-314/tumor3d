import { useEffect, useState } from "react";
import { fetchHealth } from "@viewer/api";

export function StubBanner() {
  const [backend, setBackend] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then((h) => setBackend(h.segmentation_backend))
      .catch(() => setBackend(null));
  }, []);

  if (backend !== "stub") return null;

  return (
    <div className="stub-banner" role="alert">
      <strong>Stub demo mode</strong> — API is not running real tumor AI. Results on brain MRI
      will be wrong. On RunPod:{" "}
      <code>export SEGMENTATION_BACKEND=monai</code> + MONAI bundle (see docs/runpod-setup.md).
    </div>
  );
}
