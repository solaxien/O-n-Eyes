1RUA24CSE0039,1RUA24CSE0045,1RUA24CSE0060,1RUA24CSE0062

# P3 Evaluation T38

**O(n) Eyes**

All six rows use the same 300 COCO val2017 captions and seeds at native 512 × 512 on an NVIDIA RTX 4050 Laptop GPU. Latency is warm inference through raw decode. Peak memory lists PyTorch GPU allocations and host process RSS, both in MiB. Model size counts the offline weights and configuration used in each row. The public worker also runs the safety checker.

| Configuration | CLIP similarity ↑ | FID ↓ (N = 300) | Image generation latency p50 / p95 (s) | Peak memory GPU / RSS (MiB) | Model size (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 · SD v1.4 FP32 · PNDM 50 requested / 51 calls | 0.3148 | 130.00 | 19.957 / 24.421 | 4,872 / 1,437 | 4,069 |
| 2 · SD v1.4 FP16 · PNDM 50 / 51 | 0.3148 | 129.62 | 5.993 / 6.152 | 2,677 / 1,814 | 2,035 |
| 3 · Tiny FP16 · PNDM 25 / 26 | 0.3023 | 130.89 | 1.865 / 1.927 | 1,653 / 1,795 | 1,013 |
| 4 · Tiny FP16 · DPM-Solver++ 20 / 20 | 0.3033 | 130.63 | 1.464 / 1.511 | 1,653 / 1,805 | 1,013 |
| 5 · Row 4 with TAESD | 0.3023 | 132.16 | 1.279 / 1.298 | 1,042 / 1,820 | 862 |
| 6 · Row 5 with COCO LoRA | 0.3028 | 129.16 | 1.276 / 1.291 | 1,042 / 1,840 | 864 |

## Quality versus computational cost

The tuned model was **15.6× faster** at the median than the FP32 reference. Peak GPU allocations fell **79%** and the active model bundle shrank **79%**. Host process RSS rose from 1,437 to 1,840 MiB. CLIP cosine decreased by 0.0119 and FID decreased by 0.84 within this cohort. The paired OpenCV pass took 29 ms at the median, with CLIP 0.3026 and FID 129.13 across the same 300 decodes.

On the separate 16-prompt set, Balanced rendered in 0.658 seconds at the median with CLIP 0.3387. Low resource took 0.323 seconds with CLIP 0.3234. Both profiles scored 0/16 on exact composition.

The public demo displayed a 512 px image in **2.47 seconds** in Chrome on a physical Poco F6. The laptop GPU performed inference for that request. In the final six browser runs, five displayed in 1.00–2.49 seconds and one took 11.33 seconds with 2.29 seconds inside the worker.

The separate ARM CPU comparison ran 12 native 256 px renders on the Poco F6. SD v1.4 FP16 at 25 steps took a median 218.6 seconds and 2,564 MiB peak process RSS. Tiny Q8_0 at six steps took 37.5 seconds and 761 MiB. Its bundle was 824 MiB including TAESD. The two-image prompt pair exposes substantial errors in object count, color, and position. The tested Adreno paths produced unusable outputs.

[Full ablation and methods](docs/ABLATION.md) · [Failure cases](docs/FAILURES.md) · [Fixed validation manifest](data/coco-val2017.jsonl)
