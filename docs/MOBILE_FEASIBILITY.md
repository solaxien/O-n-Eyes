# Poco F6 feasibility

The mobile target is the physical Poco F6, Android 16, four ARM CPU threads. The hosted browser route sends prompts to a laptop GPU. They have separate budgets and evidence.

| Route | Proposed acceptance budget | Status before the review |
| --- | --- | --- |
| Hosted browser | Warm submit-to-visible p95 about 10 s, native 512 px for Quality | Small browser sample had an 11.33 s outlier; no permanent host |
| Native CPU fallback | 256 px raw, warm p50 ≤45 s and p95 ≤60 s, FP16 RSS ≤1.6 GiB, bundle ≤1.1 GiB | Original two-image FP16 six-step p50 36.5 s; too few samples for stable p95 |
| Native compressed fallback | Same resolution/latency, RSS ≤1 GiB and bundle ≤0.9 GiB | Original Q8 six-step p50 37.5 s, 761 MiB RSS; slower than FP16 on that pair |
| Native quality acceptance | At least 80% count, 70% color, 50% relation and 50% exact prompt pass, confirmed by independent human review | Not achieved; these are proposed gates, not measured claims |

The native CPU budget allows a clearly labeled waiting experience. It does **not** meet a ten-second on-device interaction target. Passing pixel sanity checks is also insufficient for semantic acceptance. No finished consumer native app or full phone-side CV pipeline is claimed.

## Review measurements

The review compares newly verified base and adapter-fused FP16 exports on all 16 fixed composition prompts at six steps and native 256 px. It retains raw frames, output hashes, per-image timing, process RSS high-water marks, export/adapter/runner hashes, and battery/thermal observations. Each configuration has a separate load and warm-up. It is a larger diagnostic cohort than the earlier pair, still short of the blueprint's 30 repeated warm measurements per configuration. New numbers are reported separately in the review results, not merged into the old table.

Inference timing ends at decoded pixels inside the native runner. Host pulling, QA, optional CV, PNG writing, and 512 px display resizing are excluded and recorded separately. Host activity can delay the gap between native calls, so this run does not measure uninterrupted steady-state thermal throughput. The runner's RSS is a process lifetime high-water mark that includes load and warm-up; it is not a per-stage or whole-phone memory measure.

## Adreno failures and bounded next steps

The preserved Vulkan trace ends in the vendor shader compiler with SIGSEGV. OpenCL Flash Attention produced flat white frames. Standard attention with CPU text/decoder produced patterned noise. These are failed outputs, never fast valid generations. The CPU path is the working control. See [failure evidence](FAILURES.md).

Before another GPU model run, isolate an operator test using the same pinned GGML/runtime and current driver. Compare a small FP16 matrix multiplication and convolution against CPU reference tensors, then one U-Net step with fixed inputs. Preserve input/output hashes, max absolute/relative error, driver fingerprint, and validation-layer or crash logs. Only a passing operator/U-Net test justifies a full generation trial. This diagnostic is **planned**, not completed by the review. No driver replacement, rooting, or unsupported vendor changes are part of the fix.

If native quality or latency stays below the acceptance gates, retain the hosted UI with its execution location stated clearly and present the native CPU runner as a constrained-device research result. A substantially faster native path would need a separately validated distilled checkpoint or supported accelerator export, not just a lower step count on this checkpoint.
