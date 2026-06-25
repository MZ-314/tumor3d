# Meddollina 3D — Development Phases

Phased roadmap for the full patient-specific reconstruction pipeline: DICOM upload → preprocessing → multi-model analysis → consensus → atlas → synthetic volume → mesh → validation → interactive viewer.

**Product USP:** high-quality *predicted* 3D from as few as one MRI slice, anchored on real scan data, with measured vs AI-estimated regions clearly labeled.

**v1 clinical input:** DICOM from doctors (full series or sparse slices). Screenshot montage parsing is supported in the pipeline but secondary.

**Launch organ:** brain MRI (tumor/lesion). Platform design is organ-agnostic; knee and others follow as organ packs.

---

## Overview

| Phase | Name | Outcome |
|-------|------|---------|
| 0 | Foundation | Pipeline skeleton, schemas, async jobs, observability |
| 1 | Input intelligence | DICOM → structured `ScanContext` |
| 2 | 2D medical AI | Organ ROI, view, structures, lesion candidates |
| 3 | Consensus & anatomy | Fused 2D maps, measurements, provenance |
| 4 | Atlas & geometry | Reference anatomy warped to patient |
| 5 | Reconstruction engine | Patient-specific organ blueprint |
| 6 | Synthetic slices | Missing-slice generation with real anchors |
| 7 | Volume & mesh | NIfTI volume, organ + lesion GLB |
| 8 | Validation | Re-slice metrics, confidence scores, QA gates |
| 9 | Viewer & Meddollina | Confidence map UI, plugin, chat integration |
| 10 | Multi-organ | Knee and additional organ packs |

Phases 0–3 are prerequisites for any 3D output. Phases 4–8 deliver the core USP. Phase 9 makes it shippable in Meddollina. Phase 10 expands market.

---

## Phase 0 — Foundation

**Goal:** Replace the thin prototype spine with a pipeline that can grow without rewrites.

### Deliverables

- `backend/pipeline/reconstruct/` package layout mirroring architecture stages
- Extended shared schemas in `/shared/schemas`:
  - `ScanContext` (DICOM metadata, view, organ, slice tier, spacing)
  - `AnatomicalMap` (organ mask, lesion masks, landmarks, provenance per field)
  - `ReconstructionBlueprint` (volume shape, anchor slice indices, constraint params)
  - `ValidationReport` (Dice, IoU, SSIM per plane, overall confidence)
  - `ConfidenceRegion` (voxel/mesh regions tagged `measured` | `inference`)
- Persistent job queue (replace in-memory `_jobs` dict): job state in SQLite or Redis-backed worker
- GPU worker contract: long jobs off HTTP thread; status via `GET /reconstruct/jobs/{id}`
- Structured logging per stage (timings, model versions, failure stage)
- Config surface: model paths, atlas paths, slice-tier thresholds

### Builds on today

- FastAPI `main.py`, `reconstruct_jobs.py`, `ReconstructResponse`
- `accuracy_tier`, `ProvenanceField`, `SourceType` in `shared/schemas`

### Exit criteria

- [x] Empty pipeline runs end-to-end with stub stage outputs and valid JSON contracts
- [x] Frontend still polls jobs; no regression to chat CRUD

**Implemented:** see `backend/pipeline/reconstruct/`, `shared/schemas/pydantic/pipeline.py`, `backend/db/jobs.py`.

**Estimated effort:** 1–2 weeks

---

## Phase 1 — Input intelligence

**Goal:** Turn arbitrary uploads into a normalized `ScanContext`.

### Deliverables

- **DICOM ingest (primary path)**
  - Multi-file series upload; sort by `InstanceNumber` / `ImagePositionPatient`
  - Extract spacing, orientation, modality, body part, contrast phase hints
  - Normalize to internal volume array + affine (NIfTI-compatible)
- **Slice tier classification**
  - `single_slice` | `partial_volume` (6–9) | `multi_slice` (10+) from file count
- **Screenshot / montage path (secondary)**
  - Panel detection and split (dark-gutter heuristic + CV refinement)
  - Reject or strip PACS UI chrome, annotations, color bars
- **View detection**
  - Axial / coronal / sagittal classifier (metadata first, CNN fallback on pixels)
- **Slice position estimation**
  - Within-series: exact index from DICOM
  - Single slice: coarse z-percentile via atlas landmark regression (Phase 4 feeds this)
- **Quality gates**
  - Minimum resolution, contrast, brain-FOV checks; actionable 422 errors

### Tech

- `pydicom`, `nibabel`, `SimpleITK`, OpenCV, existing `pipeline/ingest/`

### Exit criteria

- Doctor uploads DICOM folder → `ScanContext` JSON with correct slice count, spacing, view
- Montage PNG splits into N slice images with confidence

**Estimated effort:** 2–3 weeks

---

## Phase 2 — Multi-model 2D medical AI

**Goal:** Per-slice (and mid-slice priority) organ + abnormality understanding.

### Deliverables

- **Model 1 — Organ segmentation**
  - MedSAM (promptable) or fine-tuned U-Net for brain ROI on MRI
  - Output: binary organ mask, boundary polygon, in-plane confidence
- **Model 2 — Abnormality detection**
  - Brain: MONAI BraTS bundle and/or nnU-Net lesion head on 2D slice
  - Output: lesion mask(s), bounding box, detection confidence
  - *Scope v1: brain only; knee pack later*
- **Model 3 — Medical vision analysis**
  - Structure landmarks (midline, ventricles, corpus callosum, skull boundary)
  - View/orientation verification
  - Optional VLM caption for assistant narration (not used in geometry)
- **Organ type classifier**
  - Brain vs knee vs other from DICOM tags + image classifier
  - Routes to organ pack (brain default in v1)

### Tech

- MedSAM, MONAI, nnU-Net (PyTorch), existing `pipeline/segment/`

### Exit criteria

- Single brain MRI slice → three model outputs serialized in `ModelOutputs` schema
- GPU path on RunPod; CPU stub path for dev (clearly labeled)

**Estimated effort:** 3–4 weeks

---

## Phase 3 — Consensus engine & anatomical extraction

**Goal:** One trusted 2D map + structured measurements before any 3D step.

### Deliverables

- **Consensus engine**
  - Mask fusion (STAPLE / weighted vote) where models overlap
  - Drop lesion candidates below threshold; resolve duplicate detections
  - Conflict logging for QA
- **Anatomical information extraction**
  - Organ dimensions (in-plane mm), boundary contours
  - Organ orientation (RL, AP, SI from DICOM affine)
  - Tumor: centroid, 2D bbox, boundary, relative position to landmarks
  - Distances to key structures (midline, ventricle edge) — 2D measured, depth flagged `inference`
- **Provenance on every numeric field**
  - `value`, `confidence`, `source: measured | inference`

### Builds on today

- `LesionResult`, `ProvenanceField`, `groq_assistant` summaries fed from structured output

### Exit criteria

- `AnatomicalMap` JSON with fused organ + lesion masks and measurement table
- No 3D mesh yet — 2D overlay PNG for visual QA

**Estimated effort:** 2–3 weeks

---

## Phase 4 — Atlas matching & geometric constraints

**Goal:** Reference anatomy adapted to this patient’s visible slice(s).

### Deliverables

- **Atlas library (brain v1)**
  - Template MRI + organ mask + optional structure labels
  - Versioned on disk; config via `ATLAS_BRAIN_DIR`
- **Atlas matching**
  - Register patient 2D slice(s) to atlas (SimpleITK / ANTs)
  - Similar-case retrieval optional (embedding index over reference cases)
- **Mathematical & geometric constraints**
  - Symmetry plane (brain midline)
  - Anatomical continuity along inferred z
  - Organ silhouette boundary from Phase 3 masks
  - Thickness / shell priors for single-slice depth extent
- Output: `AtlasWarpResult` (deformation field, matched z-index, constraint weights)

### Tech

- SimpleITK, ANTsPy (optional), numpy/scipy

### Exit criteria

- One brain slice → warped atlas aligned to patient ROI with visual QA slice overlay
- Registration failure returns clear error, not silent garbage

**Estimated effort:** 3–4 weeks

---

## Phase 5 — Patient-specific reconstruction engine

**Goal:** Combine real findings + atlas + constraints into an organ blueprint.

### Deliverables

- **Blueprint builder**
  - Merge measured masks on anchor slice(s)
  - Apply atlas deformation for unseen regions
  - Lock lesion voxel labels on anchor planes (no drifting tumor on real slice)
  - Organ extent in z: measured where slices exist, constrained inference elsewhere
- **Output:** `ReconstructionBlueprint` — voxel grid spec, anchor indices, label map seeds

### Exit criteria

- Blueprint renders as mid-sagittal / mid-axial preview images for internal QA
- Lesion centroid on anchor slice matches Phase 3 within tolerance

**Estimated effort:** 2–3 weeks

---

## Phase 6 — Synthetic slice generation

**Goal:** Full MRI volume where missing slices are generated, real slices preserved exactly.

### Deliverables

- **Tiered strategies**
  - `multi_slice`: minimal synthesis (gap fill between real slices, interpolation)
  - `partial_volume`: interpolation + light atlas-guided inpainting
  - `single_slice`: atlas deformation + constrained synthesis outward from anchor
- **Rules**
  - Real DICOM slices copied verbatim into volume (intensity + mask alignment)
  - Abnormality voxels on anchor planes immutable
  - Surrounding anatomy: atlas-guided; tagged `inference` in label volume
- **Output:** complete 3D intensity volume + parallel label volume (organ / lesion / confidence)

### Tech

- numpy, scipy, MONAI transforms optional; generative inpainting deferred to Phase 6b if needed

### Exit criteria

- 1-slice brain upload → synthetic volume with ≥32 slices; anchor plane bit-identical to source
- 10+ slice upload → synthetic gaps only; no replacement of real slices

**Estimated effort:** 4–6 weeks (highest research risk)

---

## Phase 7 — 3D volume reconstruction & mesh generation

**Goal:** Web-ready assets for the viewer.

### Deliverables

- **Volume assembly**
  - Intensity NIfTI + label NIfTI + confidence NIfTI
  - Voxel spacing from DICOM or defaults with provenance flag
- **Mesh generation**
  - Organ isosurface (marching cubes)
  - Lesion isosurface(s) — separate mesh per lesion
  - GLB export (Trimesh), optional OBJ/STL for download
  - Decimation for web (<100k triangles target)
- **API response**
  - Extend `ReconstructResponse`: `volume_nifti_url`, `label_nifti_url`, `confidence_nifti_url`, `organ_mesh_url`, per-lesion `mesh_url`
  - `geometry_source` per asset where mixed

### Builds on today

- `pipeline/mesh/`, `pipeline/export/nifti_export.py`, `MeshViewer`, `NiivueVolumeViewer`

### Exit criteria

- End-to-end: DICOM → GLB organ + GLB lesion + NIfTI in `/static`
- Three.js viewer loads organ + lesion toggle

**Estimated effort:** 2–3 weeks

---

## Phase 8 — Validation & accuracy verification

**Goal:** Prove consistency on real data; score inference everywhere else.

### Deliverables

- **Re-slice validation**
  - Sample generated volume at anchor plane(s)
  - Compare to uploaded MRI: SSIM (intensity), Dice/IoU (organ + lesion masks)
- **Reconstruction confidence score**
  - Aggregate: slice tier, registration quality, mask agreement, re-slice metrics
  - Per-voxel confidence map from synthesis stage
- **QA gates**
  - Fail job if anchor Dice < threshold (configurable)
  - Warn (don't fail) on single-slice volume estimates
- **Highlight AI-estimated regions**
  - Confidence volume drives viewer heatmap / mesh vertex colors

### Exit criteria

- `ValidationReport` attached to job result; visible in API and UI
- Single-slice case: high anchor-plane score; non-anchor regions marked unvalidated

**Estimated effort:** 2–3 weeks

---

## Phase 9 — Interactive viewer & Meddollina integration

**Goal:** Doctor-facing product in Meddollina chat.

### Deliverables

- **Viewer upgrades**
  - Rotate / zoom / pan (existing)
  - Toggle organ vs abnormality mesh
  - Measurements panel (2D measured, 3D estimated with badges)
  - Confidence map overlay (slice slider + 3D heatmap)
  - Download GLB/OBJ/STL
  - Prominent disclaimer + `geometry_source` banners
- **Chat UX**
  - DICOM folder upload default; slice-tier badge in progress UI
  - Assistant summary from structured `ValidationReport` + `AnatomicalMap`
- **Plugin shell**
  - `mountTumorViewer` accepts DICOM zip; modality auto from `ScanContext`
  - Embed contract documented for Meddollina team
- **Deprecate / demote**
  - TripoSR `ai_3d` → demo mode only, not clinical MRI path
  - Stub segmentation → dev-only

### Exit criteria

- Pilot-ready flow in Meddollina: upload brain DICOM → 3D + confidence + tumor toggle
- RunPod deployment doc updated

**Estimated effort:** 2–3 weeks

---

## Phase 10 — Multi-organ expansion

**Goal:** Reuse pipeline for knee and additional body parts.

### Deliverables per organ pack

- Atlas + registration priors
- Organ segmentation model (nnU-Net or MedSAM prompts)
- Abnormality model *if in scope* (knee may be structural injury, not tumor)
- Validation dataset and threshold tuning
- Modality routing in `resolve_pipeline` / organ classifier

### Suggested order

1. **Knee MRI** — volume + organ mesh; lesion optional
2. **Other MRI** — volume-only until models exist
3. **CT brain** — separate Hounsfield handling + CT atlas

### Exit criteria

- Organ classifier routes knee DICOM to knee pack; brain unchanged
- At least one non-brain organ in production with honest capability matrix

**Estimated effort:** 3–4 weeks per organ pack

---

## Dependency graph

```text
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5
                                                      │
                                                      ▼
                              Phase 9 ◄── Phase 8 ◄── Phase 7 ◄── Phase 6
                                  │
                                  ▼
                              Phase 10 (parallel after Phase 9 brain pilot)
```

Phases 6 and 4 are the critical path for single-slice USP quality.

---

## Milestones (what ships when)

| Milestone | Phases | Doctor sees |
|-----------|--------|-------------|
| **M1 — DICOM brain ingest** | 0, 1 | Upload series; 2D overlay + metadata |
| **M2 — Lesion on slice** | 2, 3 | Tumor highlighted on real slice; measurements |
| **M3 — Atlas brain 3D (1 slice)** | 4, 5, 7 | Rotatable brain mesh; lesion on anchor plane |
| **M4 — Full pipeline USP** | 6, 8 | Synthetic volume; confidence map; validation scores |
| **M5 — Meddollina pilot** | 9 | Embedded in chat; production RunPod |
| **M6 — Knee pack** | 10 | Second organ live |

---

## Infrastructure assumptions

- **GPU:** NVIDIA 6GB+ VRAM minimum; 16GB+ recommended for multi-model inference
- **Runtime:** RunPod or equivalent for pilot; async jobs mandatory
- **Storage:** `DATA_DIR` per reconstruction; NIfTI + meshes + job metadata
- **Reference data:** Licensed brain atlas + BraTS or equivalent for validation (legal review required)

---

## Risk register (by phase)

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1 | Bad DICOM metadata | Defaults + provenance flags; SimpleITK reorientation |
| 2 | Lesion false positives | Consensus thresholds; radiologist feedback loop |
| 4 | Registration fails | Fail loud; fallback to 2D-only mode |
| 6 | Synthetic slices look wrong | Anchor preservation; confidence map; tiered strategies |
| 8 | Overclaiming accuracy | Only score planes with ground truth |
| 9 | Clinical trust | Measured vs estimated UI non-negotiable |

---

## What we keep vs replace from current repo

| Keep | Replace / extend |
|------|------------------|
| FastAPI, chat SQLite, static file serving | In-memory job queue → persistent queue |
| `shared/schemas` provenance model | `ReconstructResponse` → richer assets |
| Three.js + NiiVue viewers | Confidence UI, dual mesh toggle |
| DICOM ingest basics | Full `ScanContext` pipeline |
| MONAI BraTS as Model 2 baseline | Full 3-model + consensus layer |
| TripoSR | Demo-only; not clinical MRI path |

---

## Total timeline (indicative)

| Team size | Calendar to M5 (brain pilot) |
|-----------|------------------------------|
| 2 engineers (1 ML, 1 full-stack) | ~5–7 months |
| 4 engineers (2 ML, 1 backend, 1 frontend) | ~3–4 months |

Phase 6 (synthetic slices) and Phase 4 (atlas) dominate schedule risk; start atlas data licensing early.

---

## Next action

Begin **Phase 0**: pipeline package skeleton, extended schemas, persistent jobs. Phase 1 DICOM `ScanContext` can start in parallel once schemas are frozen.
