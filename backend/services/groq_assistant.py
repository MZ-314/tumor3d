"""Groq-powered assistant narration with template fallback."""

from __future__ import annotations

import json

import httpx

from config_medical import GROQ_API_KEY, GROQ_MODEL
from shared.schemas.pydantic.reconstruct import ReconstructResponse


def _template_summary(result: ReconstructResponse, user_text: str | None) -> str:
    n = len(result.lesions)
    lesion_word = "lesion" if n == 1 else "lesions"
    tier_labels = {
        "single_slice": "single slice (depth is estimated)",
        "partial_volume": "partial volume",
        "multi_slice": "multi-slice volume",
    }
    tier = tier_labels.get(result.accuracy_tier.value, result.accuracy_tier.value)

    lines: list[str] = []

    if getattr(result, "pipeline_type", "medical") == "ai_3d":
        lines.append(
            f"Built an AI-inferred 3D mesh from your image ({result.segmentation_backend}). "
            "TripoSR treats the whole picture as one object — it does not understand MRI slices or montages."
        )
        lines.append(
            "For real brain/knee CT or MRI: use **DICOM volume** mode with your `.dcm` series. "
            "AI 3D is for a single everyday photo (one subject)."
        )
        if user_text:
            lines.insert(0, f'Re: your note — "{user_text[:120]}"')
        lines.append("Research prototype — not for clinical or metrology use.")
        return "\n\n".join(lines)

    volume_only = result.segmentation_backend == "volume_only"

    lines.extend(
        [
            f"I built a 3D volume from {result.slice_count} slice(s) ({result.modality.replace('_', ' ')}). "
            + (
                "Volume-only mode — no tumor/lesion AI on this scan."
                if volume_only
                else f"Segmentation backend: {result.segmentation_backend}."
            ),
        ]
    )

    if result.viewer_mode == "volume" and result.volume_nifti_url:
        target = "scan" if volume_only else "brain"
        if result.slice_count <= 1:
            lines.append(
                f"Only 1 slice — the 3D panel is a single MRI sheet, not a full {target}. "
                "Upload all DICOM slices from the same study (use the 📁 folder button)."
            )
        elif result.slice_count < 10:
            lines.append(
                f"{result.slice_count} slices — partial 3D stack. "
                "Upload more slices from the same series for a fuller volume."
            )
        else:
            lines.append(
                "Open the 3D panel (NiiVue): drag the crosshair to slice through axial, coronal, "
                "and sagittal planes. "
                + (
                    "Red overlay = segmentation mask (brain tumor mode only)."
                    if result.tumor_mask_nifti_url
                    else "No mask overlay — your scan volume is shown in 3D."
                )
            )

    if n == 0 and not volume_only:
        if result.viewer_mode == "volume":
            lines.append(
                "No whole-tumor region was detected by segmentation on this upload. "
                "The volume viewer still shows your scan in multiplanar 3D — there are no lesion coordinates. "
                "Upload post-contrast T1 (T1c) DICOM or 10+ axial slices for better depth and detection."
            )
        else:
            lines.append(
                "No whole-tumor region was detected by MONAI on this upload. "
                "A rotatable 3D slice preview is shown instead — there are no lesion coordinates. "
                "For tumor localization, upload post-contrast T1 (T1c) DICOM or more axial slices."
            )
    else:
        lines.append(f"Found {n} candidate {lesion_word} ({tier}).")

    for i, lesion in enumerate(result.lesions, start=1):
        c = lesion.centroid_mm
        vol = lesion.volume_mm3
        lines.append(
            f"Lesion {i} — centroid ≈ ({c.x:.1f}, {c.y:.1f}, {c.z:.1f}) mm; "
            f"volume ≈ {vol.value:.0f} mm³ "
            f"(confidence {vol.confidence:.0%}, {vol.source.value})."
        )

    lines.append(
        "Upload more axial slices to improve depth and volume estimates. "
        "This is a research prototype — not for clinical diagnosis."
    )

    if user_text:
        lines.insert(0, f"Re: your note — \"{user_text[:120]}\"")

    return "\n\n".join(lines)


async def build_assistant_summary(
    result: ReconstructResponse,
    *,
    user_text: str | None = None,
) -> str:
    if not GROQ_API_KEY:
        return _template_summary(result, user_text)

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medical imaging assistant for a research prototype. "
                    "Explain tumor localization results clearly for developers and clinicians. "
                    "Never claim diagnostic certainty. Mention confidence and that more slices improve Z."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User message: {user_text or '(none)'}\n\n"
                    f"Structured results JSON:\n{result.model_dump_json(indent=2)}"
                ),
            },
        ],
        "temperature": 0.3,
        "max_tokens": 600,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content.strip()
    except Exception:
        return _template_summary(result, user_text)
