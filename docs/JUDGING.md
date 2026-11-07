# Three-minute walkthrough and evidence map

1. Open the live URL from both a phone and laptop. Confirm the worker is ready and point out the server GPU location. Generate an unseen multi-object prompt, then show Raw/Enhanced, seed, native size, calls and elapsed time.
2. Open Benchmarks. Explain that its original COCO/device results are historical. Follow the review link for strict protocols, isolated comparisons, and the expanded native cohort. Show the phone's base/adapted comparison separately from browser latency.
3. Show one semantic failure and the color-scorer disagreement. Explain the classical repair test, the one-retry fallback policy, and why CLIP alone cannot prove correct counts or positions. End with the measured limits and deployment dependency.

Before judging, check `/api/ready`, make a real generation request from the public URL, and confirm the laptop/server and tunnel or reverse proxy will remain running. A local test or successful HTTP health response alone does not establish a usable public demo.

| Requirement | Evidence | Limit |
| --- | --- | --- |
| Real lightweight diffusion and meaningful adaptation | `p3/generate.py`, `p3/train_lora.py`, bundled LoRA and training report | Complex prompt quality remains weak |
| At least two optimization strategies | Original precision, architecture, steps, TAESD and native Q8 studies | Combined changes are labeled; new pilot isolates additional pairs |
| Classical resizing, color, edges, QA and repair | `p3/vision.py`, paired CV ablation and synthetic corruption tests | Hosted/host-side operations; cannot restore absent objects |
| Mobile resource requirements and fallback | Physical Poco F6 base/adapted native measurements and adaptive host policy | CPU-only native fallback, no ten-second native claim |
| Semantic consistency evaluation | CLIP, count/color/relation checks, 16-frame visual audit | Detector-dependent; visual review is model-assisted, not independent human annotation |
| Distribution metric | Original paired COCO FID, N=300 | Small-cohort result, not computed for N=16 pilots |
| Ablation and failure analysis | `docs/ABLATION.md`, `docs/FAILURES.md`, `docs/REVIEW_RESULTS.md` | No new training or Adreno parity study claimed |
| Phone/laptop live web UI | Responsive site and browser-generation checks | Persistent public hosting remains dependent on available infrastructure |
| Reproduction and evaluation file | Dockerfile, Python commands, pinned model revisions, `eval_P3_T38.md` | Docker runtime untested on this laptop; no unavailable organizer design template claimed |
| Interactive native Android product | Native C++ benchmark runner exists | No Gradle project, Activity, AndroidManifest or APK. The supplied brief does not explicitly require APK packaging; this remains a separate product gap |

No T4 or external GPU is used, following the user's instruction. GitHub's CPU build/check jobs are separate from GPU model evaluation.
