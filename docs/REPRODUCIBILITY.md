# Reproducing and resuming experiments

The published 300-image RTX 4050 study and original two-image Poco F6 configurations are retained as historical evidence. Their original runner did not fingerprint every input. They cannot acquire retrospective guarantees merely by hashing the files now. New measurements use protocol schema 2 in separate output directories.

## Immutable experiment identity

`protocol.json` binds the complete ordered prompt/seed manifest, model/config/tokenizer file hashes, adapter hash, precision, steps, guidance, scheduler selection, decoder, CV setting, source files, Python and package versions, CUDA/cuDNN and driver, CPU thread settings, GPU name/UUID, and measurement conventions. On Android it additionally binds the compiled runner, exported weights, decoder, OS/vendor build, backend, and export provenance.

A changed identity raises an error before replacing the old environment or samples. Choose a new `--out` directory. A changed manifest limit also creates a different experiment. The original environment stays intact. Each compatible resumed session records its own load time, warm-up, memory, and session ID.

One OS-level writer lock protects a run directory. It releases when the process dies. Images and JSON are written to unique temporary files, flushed and atomically replaced. The completed JSON record commits an image only after both display and raw PNGs have been saved. Resume checks the record's protocol, prompt, seed, status, timing, image decoding, RGB mode, dimensions, and SHA-256. A truncated JSON, changed PNG, missing raw frame, or interrupted record is regenerated. Compatible timings are kept unchanged.

Before regenerating, previous scores move into `history/<session>/scores.json`. Replaced samples are copied into that history as well. New score provenance binds the valid image/record set, paired sample IDs, scoring code, encoder revisions, detector threshold, and reference-image hashes. Reports exclude scores whose inputs changed. Derived CV runs bind their source pixels and source records. Repaired source outputs require a fresh CV experiment if their identity changes.

Old `summary.json` and the website's published evidence are snapshots, not live benchmark caches. Generate a new report explicitly after a study. Do not overwrite or relabel the published RTX study with phone or other GPU data.

## Run a controlled pilot

```sh
python -m p3.check_environment --out .runs/review-host/hardware.json
for variant in sd25 tiny25 tiny-pndm20 tiny20 taesd tuned tuned-guidance5 tuned8 tuned6; do
  python -m p3.benchmark generate --manifest data/composition.jsonl \
    --variant "$variant" --out .runs/review-host \
    --adapter p3/weights/t38-coco.safetensors || exit 1
done
python -m p3.benchmark score --manifest data/composition.jsonl \
  --run .runs/review-host/{sd25,tiny25,tiny-pndm20,tiny20,taesd,tuned,tuned-guidance5,tuned8,tuned6}
python -m p3.benchmark cv --out .runs/review-host --from-variant tuned
python -m p3.benchmark score --manifest data/composition.jsonl \
  --run .runs/review-host/tuned-{raw,full,omit-repair,omit-edges,omit-color,omit-gate}
```

The shell loop is for Bash. In PowerShell invoke each `generate` command with the desired variant. These 16 prompts are a diagnostic pilot. Their results are not a new distribution-level COCO score. The original 300-image study can be reproduced using the README's fixed COCO manifest in a new output directory.

| Pair | Intended isolated change |
| --- | --- |
| `sd25` → `tiny25` | Architecture at FP16, PNDM 25 requested steps, original VAE, no adapter |
| `tiny25` → `tiny-pndm20` | Requested steps 25 → 20 at fixed PNDM |
| `tiny-pndm20` → `tiny20` | Scheduler family, including its actual call-count convention |
| `tiny20` → `taesd` | Decoder at fixed denoising configuration |
| `taesd` → `tuned` | Fused adapter |
| `tuned` → `tuned-guidance5` | CFG 7.5 → 5.0 at a fixed 20-step budget |
| `tuned` → `tuned8` → `tuned6` | Steps at fixed resolution, decoder, guidance, and adapter |
| `tuned6` → `tuned6-384` | Native resolution only, if run |
| `tuned` raw → CV rows | Same native pixels, individual CV operations removed |

PNDM's extra call means scheduler comparison includes 21 versus 20 denoiser calls. Report both numbers. Public Balanced/Eco profiles combine CV, steps, and sometimes resolution; those are system comparisons. Sixteen samples and sequential configuration order do not establish population significance or sustained thermal behavior.

All quality metrics across multiple runs now use the same valid common IDs, even without FID. Requested, available, paired, and missing counts are separate. A detector cannot score a missing image as an ordinary composition failure. `--allow-legacy` is an explicit opt-in for rescoring historical images; it cannot reconstruct their original runtime identity.

## Verification

Training has a separate protocol that includes referenced image **contents**, actual source weights/configuration, encoding/training code, software and hardware. Changed images at the same path invalidate cached latents. Encoding tensors are checked for expected shape, FP16 dtype and finite values. Checkpoint blobs are immutable, checksum-verified files, and an atomic pointer selects the last committed optimizer/adapter/RNG state. Writer locks protect both the work directory and adapter destination. Legacy caches and existing output adapters are rejected safely. The historical adapter was not retrained or replaced during this review.

COCO downloads now verify existing PNG decoding, RGB mode and 512 × 512 dimensions. Corrupt or incomplete reference files are fetched again and replaced atomically; a valid cached image causes no network request.

```sh
python -m unittest discover -s tests -v
python -m compileall -q p3
```

The tests exercise corruption and interruption recovery, changed protocol rejection, lock exclusion, paired scoring, readiness failures, and known CV input/output properties. They use a tiny test generator for failure injection and require no model download. Real model and device measurements are reported separately.

When the pinned CLIP/YOLOS weights are already cached, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to avoid network metadata lookups during scoring. Set `HF_HOME` to the same cache used to download them. Leave offline mode unset for the first download. Interrupted scoring never commits a partial `scores.json`, and the run lock prevents a second scorer from writing concurrently.

No NVIDIA T4 or external GPU was used for the review, at the user's request. `check_environment --require-gpu "<exact torch GPU name>"` can enforce a future hardware requirement without relabeling local results.
