# P3 Ablation Study

The six main runs use the same 300 COCO val2017 captions, seeds, and 512 × 512 output size. The model is loaded once per configuration and warmed with a separate prompt. The study scores raw decodes. The public worker adds the safety check and OpenCV stages around its generated output.

| Row | Configuration | Comparison |
| --- | --- | --- |
| 1 | SD v1.4, checkpoint scheduler, 50 requested steps, FP32 | Uncompressed reference |
| 2 | SD v1.4, checkpoint scheduler, 50 requested steps, FP16 | Precision change from row 1 |
| 3 | BK-SDM-Tiny-2M, checkpoint scheduler, 25 requested steps, FP16 | Compact denoiser and fewer steps |
| 4 | BK-SDM-Tiny-2M, DPM-Solver++, 20 steps, FP16 | Scheduler and step change from row 3 |
| 5 | Row 4 with TAESD | Decoder substitution |
| 6 | Row 5 with the trained COCO LoRA | Adapter contribution |

Rows 2 to 3 change both architecture and steps. Rows 3 to 4 change both scheduler and steps. Their differences capture the combined effect. The separate Poco F6 track holds the mobile backend constant while changing model, step count, or weight format.

## Pilot

The tuned row 6 pilot completed four composition prompts on an RTX 4050 Laptop GPU. The median raw generation time was 1.266 seconds. The [failure cases](FAILURES.md) compare three actual decoded images with the corresponding OpenCV outputs. The larger COCO study is tracked separately from this pilot.

The adapter used 1,000 COCO train2017 images and 250 optimizer updates. It adds 431,616 trainable parameters. The released adapter is 1,747,488 bytes. [Training record](training.json)
