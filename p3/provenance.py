"""Content-addressed experiment protocols and crash-safe local checkpoints.

Existing experiments without a protocol are historical evidence, never resumable.
Hashing is deliberately uncached so replacing a weight at the same path is detected.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
from time import sleep
from tempfile import NamedTemporaryFile
import uuid

from PIL import Image


SCHEMA = 2
PACKAGES = ("torch", "torchvision", "diffusers", "transformers", "accelerate", "peft",
            "safetensors", "numpy", "pillow", "opencv-python-headless", "psutil", "torch-fidelity")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    atomic_bytes(path, (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def atomic_bytes(path: Path, content: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows file scanners/readers can briefly deny replacement of a closed file.
        # Retry only that sharing error, keeping the old committed file intact.
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if os.name != "nt" or attempt == 5:
                    raise
                sleep(0.02 * 2**attempt)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_image(path: Path, image: Image.Image) -> dict:
    from io import BytesIO
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    atomic_bytes(path, buffer.getvalue())
    return image_info(path)


def image_info(path: Path) -> dict:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB":
            raise ValueError("Benchmark outputs must be RGB images")
        return {"sha256": file_hash(path), "size": list(image.size), "mode": image.mode}


@contextmanager
def exclusive_run(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / ".run.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError(f"Another process is using {root}") from exc
    try:
        yield
    finally:
        stream.close()


def source_hashes(*names: str) -> dict:
    root = Path(__file__).parent
    return {name: file_hash(root / name) for name in names}


def software_identity() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


def runtime_identity(device: str) -> dict:
    import torch
    result = {"device": device, "os": platform.platform(), "machine": platform.machine(),
              "cpu": platform.processor(), "software": software_identity(),
              "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_tf32": torch.backends.cudnn.allow_tf32,
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "deterministic": torch.are_deterministic_algorithms_enabled(),
              "environment": {key: os.environ.get(key) for key in
                              ("CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                               "MKL_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF")}}
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for this experiment")
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        result["gpu"] = {"name": props.name, "uuid": str(getattr(props, "uuid", "unavailable")),
                         "bytes": props.total_memory, "capability": [props.major, props.minor]}
        try:
            result["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                text=True, timeout=15).strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            result["driver"] = "unavailable"
    return result


def establish_protocol(root: Path, protocol: dict) -> str:
    fingerprint = digest(protocol)
    path = root / "protocol.json"
    if path.exists():
        previous = read_json(path)
        if previous.get("fingerprint") != fingerprint or previous.get("protocol") != protocol:
            raise ValueError(f"Protocol changed for {root}. Use a new --out experiment directory. Existing evidence is preserved.")
    elif any(p.name != ".run.lock" for p in root.iterdir()):
        raise ValueError(f"Unverifiable legacy or incomplete experiment in {root}. Use a new --out directory.")
    else:
        write_json(path, {"fingerprint": fingerprint, "protocol": protocol})
    return fingerprint


def load_protocol(root: Path) -> dict | None:
    path = root / "protocol.json"
    if not path.exists():
        return None
    value = read_json(path)
    if digest(value["protocol"]) != value["fingerprint"]:
        raise ValueError(f"Protocol integrity check failed for {root}")
    return value


def valid_record(root: Path, sample: dict, fingerprint: str | None, *, allow_legacy=False) -> dict | None:
    try:
        record = read_json(root / "runs" / f"{sample['id']}.json")
        if record.get("sample") != sample or "error" in record:
            return None
        if fingerprint is None:
            if not allow_legacy:
                return None
            image_info(root / "images" / f"{sample['id']}.png")
            return record
        if record.get("status") != "complete" or record.get("protocol_hash") != fingerprint:
            return None
        if not record.get("artifacts") or "images" not in record["artifacts"]:
            return None
        for directory, expected in record["artifacts"].items():
            if directory not in {"images", "raw"} or image_info(root / directory / f"{sample['id']}.png") != expected:
                return None
        for name in ("generation_ms", "cv_ms"):
            if name in record and (not math.isfinite(record[name]) or record[name] < 0):
                return None
        if not any(name in record for name in ("generation_ms", "cv_ms")):
            return None
        return record
    except (OSError, ValueError, TypeError, KeyError):
        return None


def session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]


def invalidate_scores(root: Path, session: str):
    path = root / "scores.json"
    if path.exists():
        archive = root / "history" / session / "scores.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, archive)


def archive_sample(root: Path, key: str, session: str):
    for directory, suffix in (("runs", ".json"), ("images", ".png"), ("raw", ".png")):
        path = root / directory / f"{key}{suffix}"
        if path.exists():
            archive = root / "history" / session / directory / path.name
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, archive)


def inputs_snapshot(root: Path, rows: list[dict], *, allow_legacy=False) -> dict:
    protocol = load_protocol(root)
    fingerprint = protocol["fingerprint"] if protocol else None
    items = {}
    for sample in rows:
        record = valid_record(root, sample, fingerprint, allow_legacy=allow_legacy)
        if record is not None:
            items[sample["id"]] = {"sample": digest(sample),
                                   "image": file_hash(root / "images" / f"{sample['id']}.png"),
                                   "record": digest(record)}
    return {"protocol_hash": fingerprint, "items": items}


def scores_current(root: Path, scores: dict) -> bool:
    try:
        proof = scores["provenance"]
        if proof["scorer_code"] != source_hashes("score.py", "provenance.py"):
            return False
        if proof["inputs"] != inputs_snapshot(root, proof["manifest"], allow_legacy=proof["legacy"]):
            return False
        return all(file_hash(Path(row["path"])) == row["sha256"] for row in proof.get("references", []))
    except (OSError, ValueError, KeyError, TypeError):
        return False
