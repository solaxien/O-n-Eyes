# Handoff: v2 improvement plan

Branch: `v2-verifier`, built on `v2`. This file says what was changed, what was verified, and the exact steps only the team can do (the phone, the GitHub release and the deployment).

## 1. Status by phase

| Phase | Code | Verified here | Team still needs to |
| --- | --- | --- | --- |
| 1. Trusted baseline | `scripts/run_phase1.ps1` and `scripts/phase1_report.py` | Report generator and KID unit-tested | Run the phone study and paste the table into `eval_P3_T38.md` |
| 2. On-device verifier | `browser/src/verifier.js`, worker integration, UI checks | 32 browser tests; JS detector output matches Python ONNX on two real phone images | Prepare the detector (step 2) and time it on the phone |
| 3. Retry and selection | `browser/src/selection.js`, candidate loop in `app.js`, timeline UI | Policy tests | Run `-SelectionStudy` to show the gain on the offline scorer |
| 4. INT8 | `p3/quantize_browser.py` | Toy-model test (MatMulNBits, sharding, agreement report) | Run on the real export, test desktop WebGPU, then the phone; keep or reject |
| 5. Classical CV evidence | Exposure stage, per-stage reasons and timing, defect report, offline BRISQUE | Unit tests; BRISQUE smoke test | Run `scripts/offline_iqa.py` (included in the Phase 1 script) |
| 6. Android hardening | Readiness report, simulated budgets, collapsible sections, installable app | Planner and readiness tests; every UI element id checked | Physical checks in step 5 |

Not delivered: an on-device image-text alignment model (no compact CLIP-style model could be fetched in the build environment). The verifier reports alignment as not bundled.

## 2. Cache the verifier on the device

Use Prepare while online. The packaged site and Python host serve the official YOLOX-Nano release through `/models/verifier-yolox-nano.onnx`. The browser checks the size and SHA-256 pinned in `browser/verifier-manifest.json` before caching it. Existing installations can fetch the verifier while reusing their generation weights. For a fully offline local host, place the matching detector at `.models/browser/v3/verifier-yolox-nano.onnx` before opening the site.

The setup panel reports when prompt checks are available offline. If the download fails, generation remains available and the panel offers a retry.

## 3. Build, test and deploy

```sh
cd browser
npm ci
npm test            # set P3_VERIFIER_MODEL=<path to the onnx> to also run the parity test
npm run build
cd ..
python scripts/serve_device.py --port 7860 --remote-models https://github.com/solaxien/O-n-Eyes/releases/download/browser-models-v3
```

Deploy as before with `scripts/package_browser_site.mjs`; it now routes the verifier file and embeds the app icons. On a phone that already has the models, press Prepare once so the verifier is cached.

## 4. Phase 1 measurements (phone)

Close other Chrome tabs, keep the phone on USB power, note the battery temperature, then:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase1.ps1 -Origin <https-origin> -Tab <tab-id> -Adb <adb.exe> -SelectionStudy
```

Output: `report/phase1-results.md` (the table for `eval_P3_T38.md`), `cost.json`, `scores.json` per run, `iqa-*.json`, and a selection study. Expect roughly 30 to 60 minutes. Once the new table is in the eval file, delete the two "historical v3" banners in `eval_P3_T38.md` and `docs/BROWSER_ON_DEVICE.md`.

## 5. Physical Android checks to record

- Offline reload and generation with Wi-Fi and data off.
- Storage eviction: clear site data, confirm Prepare repairs the cache.
- Tab suspension: switch apps for two minutes, return, generate.
- Low battery: one generation below 20%, note time and temperature.
- Device budget menu: "Simulate a 4 GB phone" shows a lighter Auto plan; "Simulate a memory failure" shows the fallback chain in the run report.
- Install to home screen and open offline.

## 6. INT8 experiment (optional)

```sh
pip install -r requirements-web-export.txt
python -m p3.quantize_browser --export .models/browser/v3 --out .models/browser/v3-int8
```

Read `quantize-report.json` (size and agreement). Build with `P3_BROWSER_MODELS=.models/browser/v3-int8`, test on desktop WebGPU, then the phone. Keep it only if WebGPU runs `MatMulNBits` correctly and storage or memory improves without a visible quality loss. Either outcome goes in the eval file.

## 7. Demo script (3 minutes)

1. Open the app on the phone, show the readiness report.
2. Switch on airplane mode.
3. Generate a judge's prompt in Auto; show the time and Raw versus Enhanced.
4. Generate a two-object prompt in Balanced; show the prompt checks and, if a clause failed, the retry.
5. Set Device budget to "Simulate a memory failure" and generate; show the fallback in the run report.
6. Open the Optimization lab: 20-step reference versus the selected profile, same prompt and seed.

## 8. Tests

- Browser: `cd browser && npm test` (32 tests; the parity test runs when `P3_VERIFIER_MODEL` is set).
- Python: `python -m unittest discover -s tests` (24 tests; needs PyTorch and the packages in CI).
- CI also builds `Dockerfile.device`.
