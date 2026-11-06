1RUA24CSE0039,1RUA24CSE0045,1RUA24CSE0060,1RUA24CSE0062

# P3 Evaluation T38

**O(n) Eyes**

Status of this snapshot · The four-prompt GPU pilot is measured. The six 300-image runs and Poco F6 comparison are in progress.

## CLIP similarity

The four composition pilot prompts averaged 0.3581 cosine for the tuned raw images and 0.3586 after OpenCV. The held-out COCO study will report CLIP for all six rows.

## FID

The evaluation uses 300 matched COCO val2017 references, with the sample size reported beside every score. The full FID run is in progress.

## Image generation latency

On the RTX 4050 Laptop GPU, the tuned 20-step pilot took a median 1.266 seconds per raw decode at 512 × 512 over four warm runs. Browser and native phone latency are measured separately.

## Peak memory consumption

The tuned GPU pilot peaked at 1,042 MiB of allocated GPU memory and 1,808 MiB of process RSS. The phone runner records the process high-water RSS.

## Model size

The trained COCO LoRA is 1,747,488 bytes. The Tiny mobile FP16 checkpoint is 1,060,307,888 bytes and its Q8_0 export is 853,972,656 bytes. The host study counts the weights and configuration used in each row.

## Quality versus computational cost

The [ablation study](docs/ABLATION.md) lists the six matched configurations. The [failure case analysis](docs/FAILURES.md) shows three semantic errors that persist after the OpenCV pass.
