"""SQLite persistence for async reconstruction jobs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from config_medical import DB_PATH, ensure_data_dirs
from shared.schemas.pydantic.reconstruct import ReconstructResponse

JobStatus = Literal["processing", "done", "error"]


@dataclass
class ReconstructJob:
    status: JobStatus = "processing"
    result: ReconstructResponse | None = None
    error: str | None = None
    slice_count: int = 0
    pipeline: str = "medical"
    stage: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    ensure_data_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_jobs_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS reconstruct_jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                pipeline TEXT NOT NULL,
                slice_count INTEGER NOT NULL,
                modality TEXT,
                chat_id TEXT,
                stage TEXT,
                error TEXT,
                result_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reconstruct_jobs_status
                ON reconstruct_jobs(status);
            """
        )


def create_job_record(
    job_id: str,
    *,
    pipeline: str,
    slice_count: int,
    modality: str,
    chat_id: str | None = None,
) -> None:
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO reconstruct_jobs (
                id, status, pipeline, slice_count, modality, chat_id,
                stage, error, result_json, created_at, updated_at
            ) VALUES (?, 'processing', ?, ?, ?, ?, 'queued', NULL, NULL, ?, ?)
            """,
            (job_id, pipeline, slice_count, modality, chat_id, now, now),
        )


def update_job_stage(job_id: str, stage: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE reconstruct_jobs SET stage = ?, updated_at = ? WHERE id = ?",
            (stage, _now(), job_id),
        )


def complete_job(job_id: str, result: ReconstructResponse) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE reconstruct_jobs
            SET status = 'done', stage = 'complete', result_json = ?, error = NULL, updated_at = ?
            WHERE id = ?
            """,
            (result.model_dump_json(), _now(), job_id),
        )


def fail_job(job_id: str, error: str, *, stage: str | None = None) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE reconstruct_jobs
            SET status = 'error', error = ?, stage = COALESCE(?, stage), updated_at = ?
            WHERE id = ?
            """,
            (error, stage, _now(), job_id),
        )


def get_job_record(job_id: str) -> ReconstructJob | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT status, pipeline, slice_count, stage, error, result_json
            FROM reconstruct_jobs WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
    if not row:
        return None

    result = None
    if row["result_json"]:
        result = ReconstructResponse.model_validate_json(row["result_json"])

    return ReconstructJob(
        status=row["status"],
        result=result,
        error=row["error"],
        slice_count=row["slice_count"],
        pipeline=row["pipeline"],
        stage=row["stage"],
    )
