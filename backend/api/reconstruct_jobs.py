"""In-memory async reconstruction jobs (avoids RunPod HTTP proxy timeouts)."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from api.pipeline_routing import PIPELINE_AI_3D
from config_medical import MedicalPipelineError
from config_reconstruction import Image3DError
from db.database import add_message, get_chat, touch_chat
from image_to_3d_pipeline import process_image_to_3d
from medical_pipeline import process_medical_slices
from shared.schemas.pydantic.reconstruct import ReconstructResponse

logger = logging.getLogger(__name__)

ASYNC_SLICE_THRESHOLD = int(os.environ.get("ASYNC_SLICE_THRESHOLD", "3"))

JobStatus = Literal["processing", "done", "error"]


@dataclass
class ReconstructJob:
    status: JobStatus = "processing"
    result: ReconstructResponse | None = None
    error: str | None = None
    slice_count: int = 0
    pipeline: str = "medical"


_jobs: dict[str, ReconstructJob] = {}


def should_run_async(pipeline: str, slice_count: int) -> bool:
    if pipeline == PIPELINE_AI_3D:
        return True
    return slice_count >= ASYNC_SLICE_THRESHOLD


def get_job(job_id: str) -> ReconstructJob | None:
    return _jobs.get(job_id)


def start_job(
    job_id: str,
    *,
    pipeline: str,
    slice_paths: list[Path],
    work_dir: Path,
    modality: str,
    chat_id: str | None,
    user_text: str | None,
    upload_label: str,
    first_filename: str,
) -> None:
    _jobs[job_id] = ReconstructJob(
        status="processing",
        slice_count=len(slice_paths),
        pipeline=pipeline,
    )
    asyncio.create_task(
        _run_job(
            job_id,
            pipeline=pipeline,
            slice_paths=slice_paths,
            work_dir=work_dir,
            modality=modality,
            chat_id=chat_id,
            user_text=user_text,
            upload_label=upload_label,
            first_filename=first_filename,
        )
    )


async def _run_job(
    job_id: str,
    *,
    pipeline: str,
    slice_paths: list[Path],
    work_dir: Path,
    modality: str,
    chat_id: str | None,
    user_text: str | None,
    upload_label: str,
    first_filename: str,
) -> None:
    job = _jobs[job_id]
    logger.info(
        "Job %s started — pipeline=%s, %d file(s), modality=%s",
        job_id,
        pipeline,
        len(slice_paths),
        modality,
    )
    try:
        if pipeline == PIPELINE_AI_3D:
            result = await process_image_to_3d(
                slice_paths[0],
                work_dir,
                chat_id=chat_id,
                user_text=user_text,
            )
        else:
            result = await process_medical_slices(
                slice_paths,
                work_dir,
                modality=modality,
                chat_id=chat_id,
                user_text=user_text,
            )
        if chat_id:
            detail = get_chat(chat_id)
            if detail:
                title = f"{modality}: {first_filename}"[:80]
                if detail.title == "New scan":
                    touch_chat(chat_id, title=title)
                add_message(
                    chat_id,
                    "user",
                    text=user_text or upload_label,
                    attachment_url=result.source_image_url,
                )
                add_message(
                    chat_id,
                    "assistant",
                    text=result.assistant_summary,
                    reconstruction=result,
                )
                result.chat_id = chat_id

        job.status = "done"
        job.result = result
        logger.info("Job %s finished — pipeline=%s", job_id, pipeline)
    except (MedicalPipelineError, Image3DError) as exc:
        job.status = "error"
        job.error = str(exc)
        logger.warning("Job %s failed: %s", job_id, exc)
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        logger.exception("Job %s unexpected error", job_id)
