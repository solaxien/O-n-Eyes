"""Record real hardware before a study; optionally enforce an exact GPU name."""

import argparse
from pathlib import Path

from p3.provenance import runtime_identity, write_json


def verify_gpu(identity: dict, required: str | None):
    if required and identity.get("gpu", {}).get("name") != required:
        raise ValueError(f"Required {required!r}; actual GPU is {identity.get('gpu', {}).get('name')!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-gpu", help="Exact torch device name, never a host label supplied as evidence.")
    args = parser.parse_args()
    identity = runtime_identity(args.device)
    verify_gpu(identity, args.require_gpu)
    write_json(args.out, identity)
    print(identity.get("gpu", {"name": "CPU"}))


if __name__ == "__main__":
    main()
