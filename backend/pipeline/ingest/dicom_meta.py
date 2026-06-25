"""DICOM metadata extraction for ScanContext (Phase 1)."""

from __future__ import annotations

from pathlib import Path

from shared.schemas.pydantic.common import SourceType
from shared.schemas.pydantic.pipeline import MriView, OrganType, SliceSpacing


def _is_dicom(path: Path) -> bool:
    return path.suffix.lower() in {".dcm", ".dicom"}


def detect_input_source(paths: list[Path]) -> str:
    if paths and all(_is_dicom(p) for p in paths):
        return "dicom"
    return "image"


def detect_mri_view_from_dicom(path: Path) -> MriView:
    import pydicom

    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    iop = getattr(ds, "ImageOrientationPatient", None)
    if not iop or len(iop) != 6:
        return MriView.UNKNOWN

    row = [float(iop[0]), float(iop[1]), float(iop[2])]
    col = [float(iop[3]), float(iop[4]), float(iop[5])]
    normal = [
        row[1] * col[2] - row[2] * col[1],
        row[2] * col[0] - row[0] * col[2],
        row[0] * col[1] - row[1] * col[0],
    ]
    ax = abs(normal[0])
    ay = abs(normal[1])
    az = abs(normal[2])
    if az >= ax and az >= ay:
        return MriView.AXIAL
    if ay >= ax and ay >= az:
        return MriView.CORONAL
    if ax >= ay and ax >= az:
        return MriView.SAGITTAL
    return MriView.UNKNOWN


def organ_type_from_dicom(path: Path, modality: str) -> OrganType:
    import pydicom

    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    body = str(getattr(ds, "BodyPartExamined", "") or "").lower()
    desc = str(getattr(ds, "SeriesDescription", "") or "").lower()
    combined = f"{body} {desc} {modality.lower()}"

    if "brain" in combined or "head" in combined or "cran" in combined:
        return OrganType.BRAIN
    if "knee" in combined:
        return OrganType.KNEE
    if modality.lower().startswith("brain"):
        return OrganType.BRAIN
    if modality.lower().startswith("knee"):
        return OrganType.KNEE
    return OrganType.OTHER


def slice_spacing_from_dicom(path: Path) -> SliceSpacing:
    import pydicom

    from config_medical import DEFAULT_PIXEL_SPACING_MM, DEFAULT_SLICE_THICKNESS_MM

    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    row_sp = col_sp = DEFAULT_PIXEL_SPACING_MM
    thickness = DEFAULT_SLICE_THICKNESS_MM
    source = SourceType.INFERENCE

    if hasattr(ds, "PixelSpacing") and ds.PixelSpacing is not None:
        row_sp = float(ds.PixelSpacing[0])
        col_sp = float(ds.PixelSpacing[1])
        source = SourceType.MEASURED
    if hasattr(ds, "SliceThickness") and ds.SliceThickness:
        thickness = float(ds.SliceThickness)
        source = SourceType.MEASURED

    return SliceSpacing(
        row_mm=row_sp,
        col_mm=col_sp,
        slice_mm=thickness,
        source=source,
    )
