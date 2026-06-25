import type { AnatomicalModule } from "@shared/index";

const SOURCE_LABELS: Record<string, string> = {
  measured: "Measured (anchor)",
  inference: "Atlas / AI inference",
};

interface AnatomicalExplorerPanelProps {
  modules: AnatomicalModule[];
  visibleModules: Set<string>;
  onToggle: (moduleId: string) => void;
  onShowAll?: () => void;
  onHideAll?: () => void;
}

function sourceLabel(mod: AnatomicalModule): string {
  if (mod.anchor_locked) return SOURCE_LABELS.measured;
  return SOURCE_LABELS[mod.geometry_source] ?? mod.geometry_source;
}

function sourceClass(mod: AnatomicalModule): string {
  if (mod.anchor_locked || mod.geometry_source === "measured")
    return "modular-explorer__badge--measured";
  if (mod.morph_applied) return "modular-explorer__badge--morphed";
  return "modular-explorer__badge--inference";
}

export function AnatomicalExplorerPanel({
  modules,
  visibleModules,
  onToggle,
  onShowAll,
  onHideAll,
}: AnatomicalExplorerPanelProps) {
  if (modules.length === 0) return null;

  const morphed = modules.filter((m) => m.morph_applied);
  const measured = modules.filter((m) => m.anchor_locked || m.geometry_source === "measured");

  return (
    <div className="modular-explorer">
      <div className="modular-explorer__header">
        <h3 className="modular-explorer__title">Advanced Anatomical Explorer</h3>
        <div className="modular-explorer__actions">
          {onShowAll && (
            <button type="button" className="modular-explorer__btn" onClick={onShowAll}>
              Show all
            </button>
          )}
          {onHideAll && (
            <button type="button" className="modular-explorer__btn" onClick={onHideAll}>
              Hide all
            </button>
          )}
        </div>
      </div>

      <div className="modular-explorer__legend" aria-label="Provenance legend">
        <span className="modular-explorer__legend-item modular-explorer__legend-item--measured">
          Measured
        </span>
        <span className="modular-explorer__legend-item modular-explorer__legend-item--inference">
          Atlas / AI
        </span>
        <span className="modular-explorer__legend-item modular-explorer__legend-item--morphed">
          Morphed
        </span>
      </div>

      {morphed.length > 0 && (
        <p className="modular-explorer__audit">
          <strong>Morph audit:</strong>{" "}
          {morphed.map((m) => m.display_name).join(", ")}
          {measured.length > 0 && " · anchor/tumor unchanged"}
        </p>
      )}

      <ul className="modular-explorer__list">
        {modules.map((mod) => (
          <li key={mod.module_id} className="modular-explorer__item">
            <label className="modular-explorer__label">
              <input
                type="checkbox"
                checked={visibleModules.has(mod.module_id)}
                onChange={() => onToggle(mod.module_id)}
              />
              <span className="modular-explorer__name">{mod.display_name}</span>
            </label>
            <span className={`modular-explorer__badge ${sourceClass(mod)}`}>
              {sourceLabel(mod)}
              {mod.morph_applied ? " · morphed" : ""}
            </span>
            <span className="modular-explorer__conf">
              {Math.round(mod.confidence * 100)}%
              {mod.connects_to.length > 0 && (
                <span className="modular-explorer__links" title={mod.connects_to.join(", ")}>
                  {" "}
                  · {mod.connects_to.length} links
                </span>
              )}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
