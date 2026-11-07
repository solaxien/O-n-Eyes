"""One persistent inference worker serving a responsive site and a bounded job API."""

import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
import gc
from io import BytesIO
import logging
import math
import os
from pathlib import Path
import secrets
from threading import Lock
from time import perf_counter
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints
import torch

from p3.generate import GenerationError, Generator, PROFILES, default_device, pick_profile
from p3.vision import resize


LOG = logging.getLogger("p3")
MAX_PENDING = 2
MAX_RETAINED = 8
RESULT_TTL = 180


class ImageRequest(BaseModel):
    prompt: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=1000)]
    mode: Literal["auto", "balanced", "quality", "eco"] = "auto"
    seed: int | None = Field(default=None, ge=0, le=4_294_967_295)


@dataclass
class Job:
    created: float
    status: str = "queued"
    stage: str = "Waiting for the worker"
    step: int = 0
    total: int = 0
    finished: float | None = None
    result: dict | None = None
    error: str | None = None

    def response(self) -> dict:
        return {
            "status": self.status,
            "stage": self.stage,
            "step": self.step,
            "total": self.total,
            "result": self.result,
            "error": self.error,
        }


def _webp(image) -> str:
    buffer = BytesIO()
    image.save(buffer, "WEBP", quality=91, method=3)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class Worker:
    def __init__(self):
        self.lock = Lock()
        self.jobs: dict[str, Job] = {}
        self.generator: Generator | None = None
        self.state = "loading"
        self.estimates = {"balanced": 5.5, "quality": 7.5, "eco": 3.5, "cpu": 90.0}
        self.quality_retry_after = 0.0
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diffusion")
        self.executor.submit(self._load)

    def _load(self):
        try:
            device = default_device()
            decoder = os.environ.get("P3_DECODER", "taesd")
            adapter_path = os.environ.get("P3_ADAPTER")
            bundled_adapter = Path(__file__).with_name("weights") / "t38-coco.safetensors"
            adapter = Path(adapter_path) if adapter_path else bundled_adapter if bundled_adapter.is_file() else None
            generator = None
            if device == "cuda":
                try:
                    generator = Generator(decoder=decoder, adapter=adapter)
                except torch.OutOfMemoryError:
                    LOG.warning("Insufficient GPU memory to load the model. Loading CPU fallback.")
                if generator is None:
                    gc.collect()
                    torch.cuda.empty_cache()
            if generator is None:
                generator = Generator(
                    device="cpu", precision="fp32", weight_precision="fp16", decoder=decoder, adapter=adapter
                )
            budget = float(os.environ.get("P3_REQUEST_BUDGET", "10" if generator.device == "cuda" else "180"))
            if not 1 <= budget <= 600:
                raise ValueError("P3_REQUEST_BUDGET must be between 1 and 600 seconds")
            if generator.device == "cuda":
                try:
                    warmup = generator.generate(
                        "A photo of a green pear on a kitchen counter.", 0, PROFILES["quality"]
                    )
                    self.estimates["quality"] = 1.3 * warmup.metadata["generation_ms"] / 1000 + 0.35
                except (GenerationError, torch.OutOfMemoryError):
                    self._defer_quality()
                    torch.cuda.empty_cache()
            with self.lock:
                self.generator = generator
                self.budget = budget
                self.state = "ready"
            LOG.info("Ready on %s", generator.hardware)
        except Exception:
            LOG.exception("The model worker could not start")
            with self.lock:
                self.state = "failed"

    def health(self) -> dict:
        with self.lock:
            return {
                "status": self.state,
                "hardware": self.generator.hardware if self.generator else None,
                "backend": self.generator.device if self.generator else None,
            }

    def _defer_quality(self):
        # Auto uses smaller profiles briefly; explicit Quality can still retry
        # when memory and a conservative finite estimate fit its budget.
        previous = self.estimates["quality"]
        self.estimates["quality"] = max(7.5, previous if math.isfinite(previous) else 0.0)
        self.quality_retry_after = perf_counter() + 30.0

    def _observe(self, key: str, elapsed: float):
        estimate = self.estimates[key]
        if not math.isfinite(estimate):
            estimate = 1.25 * elapsed
        self.estimates[key] = max(0.35, 0.7 * estimate + 0.3 * 1.25 * elapsed)
        if key == "quality":
            self.quality_retry_after = 0.0

    def submit(self, request: ImageRequest) -> str:
        with self.lock:
            if self.state == "loading":
                raise HTTPException(503, "Model weights are loading. Try again when the worker is ready.")
            if self.state == "failed":
                raise HTTPException(503, "The worker could not start. Ask the host to check its logs.")
            now = perf_counter()
            finished = sorted((job.finished, key) for key, job in self.jobs.items() if job.finished is not None)
            for time, key in finished:
                if now - time > RESULT_TTL or len(self.jobs) >= MAX_RETAINED:
                    del self.jobs[key]
            if sum(job.finished is None for job in self.jobs.values()) >= MAX_PENDING:
                raise HTTPException(429, "The GPU queue is full. Try again in a few seconds.")
            job_id = secrets.token_urlsafe(18)
            job = Job(created=now)
            self.jobs[job_id] = job
            self.executor.submit(self._run, job, request)
        return job_id

    def read(self, job_id: str) -> dict:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or (job.finished is not None and perf_counter() - job.finished > RESULT_TTL):
                raise HTTPException(404, "This run has expired. Generate another image.")
            return job.response()

    def _run(self, job: Job, request: ImageRequest):
        generator = self.generator
        assert generator is not None
        started = perf_counter()
        deadline = job.created + self.budget

        def remaining() -> float:
            return deadline - perf_counter()

        def progress(stage: str, step: int, total: int):
            if remaining() <= 0:
                raise GenerationError("This run reached its time limit. Try Low resource mode.")
            with self.lock:
                job.status = "running"
                job.stage, job.step, job.total = stage, step, total

        try:
            mode = request.mode
            if mode == "auto" and perf_counter() < self.quality_retry_after:
                mode = "balanced"
            key, reason = pick_profile(mode, generator.device, generator.free_bytes(), remaining(), self.estimates.__getitem__)
            seed = request.seed if request.seed is not None else secrets.randbits(32)
            attempts = 1
            result = None
            try:
                result = generator.generate(request.prompt, seed, PROFILES[key], progress=progress)
            except torch.OutOfMemoryError:
                LOG.warning("GPU memory exhausted during %s generation", key)
                if key == "quality":
                    self._defer_quality()
            if result is None:
                gc.collect()
                if generator.device == "cuda":
                    torch.cuda.empty_cache()
                if generator.device != "cuda" or key == "eco" or remaining() < self.estimates["eco"]:
                    raise GenerationError("This run exceeded available memory. Try again in Low resource mode.")
                key = "eco"
                reason = "Retried at lower resolution after a GPU memory limit."
                attempts += 1
                result = generator.generate(request.prompt, seed, PROFILES[key], progress=progress)
            if not result.metadata["cv"]["usable"]:
                if attempts == 1 and remaining() >= self.estimates[key]:
                    seed = (seed + 1) % 2**32
                    reason = "Retried with the next seed after the image quality check."
                    attempts += 1
                    result = generator.generate(request.prompt, seed, PROFILES[key], progress=progress)
                if not result.metadata["cv"]["usable"]:
                    raise GenerationError("The quality check flagged the result. Try another seed or Quality mode.")
            elapsed = result.metadata["generation_ms"] / 1000
            self._observe(key, elapsed)
            progress("Packaging image", result.metadata["steps"], result.metadata["steps"])
            raw = _webp(resize(result.raw, 512))
            enhanced = _webp(result.image)
            metadata = result.metadata | {
                "queue_ms": round((started - job.created) * 1000, 2),
                "server_ms": round((perf_counter() - job.created) * 1000, 2),
                "attempts": attempts,
                "selection_reason": reason,
            }
            with self.lock:
                job.result = {"raw": raw, "enhanced": enhanced, "metadata": metadata}
                job.stage, job.status, job.finished = "Complete", "complete", perf_counter()
        except (GenerationError, torch.OutOfMemoryError) as exc:
            message = str(exc) if isinstance(exc, GenerationError) else "Available memory was exhausted. Try again later."
            self._fail(job, message)
        except Exception:
            LOG.exception("Generation failed")
            self._fail(job, "Generation failed. Retry or ask the host to check its logs.")

    def _fail(self, job: Job, message: str):
        with self.lock:
            job.stage, job.status, job.finished, job.error = "Generation stopped", "failed", perf_counter(), message


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.worker = Worker()
    try:
        yield
    finally:
        application.state.worker.executor.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title="P3 Image Lab", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/api/health")
def health(request: Request):
    return request.app.state.worker.health()


@app.get("/api/ready")
def ready(request: Request):
    state = request.app.state.worker.health()
    if state["status"] != "ready":
        raise HTTPException(503, state)
    return state


@app.post("/api/jobs", status_code=202)
def start_job(payload: ImageRequest, request: Request):
    return {"id": request.app.state.worker.submit(payload)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    return request.app.state.worker.read(job_id)


app.mount("/", StaticFiles(directory=Path(__file__).with_name("web"), html=True), name="web")
