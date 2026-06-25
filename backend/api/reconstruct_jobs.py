"""In-memory async reconstruction jobs (avoids RunPod HTTP proxy timeouts)."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from api.pipeline_routing import PIPELINE_AI_3D, PIPELINE_MEDICAL_TUMOR
from config_medical import MedicalPipelineError
from config_reconstruction import Image3DError
from db.database import add_message, get_chat, touch_chat
from db.jobs import (
    ReconstructJob,
    complete_job,
    create_job_record,
    fail_job,
    get_job_record,
    update_job_stage,
)
from image_to_3d_pipeline import process_image_to_3d
from pipeline.reconstruct import run_reconstruction_pipeline
from shared.schemas.pydantic.reconstruct import ReconstructResponse

logger = logging.getLogger(__name__)

ASYNC_SLICE_THRESHOLD = int(os.environ.get("ASYNC_SLICE_THRESHOLD", "3"))

JobStatus = Literal["processing", "done", "error"]


def should_run_async(pipeline: str, slice_count: int) -> bool:
    if pipeline in (PIPELINE_AI_3D, PIPELINE_MEDICAL_TUMOR):
        return True
    return slice_count >= ASYNC_SLICE_THRESHOLD


def get_job(job_id: str) -> ReconstructJob | None:
    return get_job_record(job_id)


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
    create_job_record(
        job_id,
        pipeline=pipeline,
        slice_count=len(slice_paths),
        modality=modality,
        chat_id=chat_id,
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
    logger.info(
        "Job %s started — pipeline=%s, %d file(s), modality=%s",
        job_id,
        pipeline,
        len(slice_paths),
        modality,
    )

    def on_stage(stage: str) -> None:
        update_job_stage(job_id, stage)

    try:
        if pipeline == PIPELINE_AI_3D:
            on_stage("ai_3d")
            result = await process_image_to_3d(
                slice_paths[0],
                work_dir,
                chat_id=chat_id,
                user_text=user_text,
            )
        else:
            result = await run_reconstruction_pipeline(
                slice_paths,
                work_dir,
                modality=modality,
                chat_id=chat_id,
                user_text=user_text,
                on_stage=on_stage,
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

        complete_job(job_id, result)
        logger.info("Job %s finished — pipeline=%s", job_id, pipeline)
    except (MedicalPipelineError, Image3DError) as exc:
        fail_job(job_id, str(exc))
        logger.warning("Job %s failed: %s", job_id, exc)
    except Exception as exc:
        fail_job(job_id, str(exc))
        logger.exception("Job %s unexpected error", job_id)
