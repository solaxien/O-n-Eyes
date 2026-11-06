# P3 Failure Case Analysis

These examples use the tuned BK-SDM-Tiny-2M checkpoint, the COCO LoRA, TAESD, and 20 DPM-Solver++ calls at 512 × 512. They come from a four-prompt composition pilot separate from COCO val2017. The pilot averaged 0.3581 CLIP cosine before CV and 0.3586 after CV. All four prompts failed at least one automatic count, color, or position check.

| Case | Prompt and seed | Visible failure | Effect of CV |
| --- | --- | --- | --- |
| Position and color | One blue cup to the left of one red apple on a white tabletop, seed 38 | A red apple stays on the left, a second apple sits inside a white cup, and the blue appears in the tabletop covering. Swapping the object order in the prompt preserved much of the layout. | Speckle repair and edge filtering leave the relative positions and cup color as generated. [Raw](../p3/web/evidence/blue-cup-left-raw.webp) · [Enhanced](../p3/web/evidence/blue-cup-left-cv.webp) |
| Object class and count | Exactly two yellow bananas beside one red apple, seed 841 | Several round yellow fruits appear. Their shapes and count fail the banana clause even though the red fruit and yellow color are visible. | The filters repair isolated pixels. The object count and shapes persist. [Raw](../p3/web/evidence/two-bananas-raw.webp) · [Enhanced](../p3/web/evidence/two-bananas-cv.webp) |
| Attribute binding and geometry | A blue umbrella above a red bench with a green backpack, seed 72 | The output includes a blue umbrella, a second red umbrella, a teal bench, and a red bag. Some bench supports break into small fragments. | Local repair and denoising preserve the scene and its attribute errors. [Raw](../p3/web/evidence/bench-rain-raw.webp) · [Enhanced](../p3/web/evidence/bench-rain-cv.webp) |

The CLIP mean captures scene-level alignment. The stricter checks expose where the scene loses individual constraints. The runtime quality gate measures pixel integrity and can retry a near-blank or heavily corrupted decode. Reversed relations, duplicated objects, and color swaps can pass that gate. In this pilot the CV stage made local pixel changes without resolving the semantic failures.

## Poco F6 backend failures

The Adreno Vulkan build stopped in the vendor shader compiler with SIGSEGV. OpenCL with Flash Attention returned eight solid-white frames across four matched configurations. With standard attention and CPU TAESD, the 25-step hybrid run returned two frames with repeating colored patterns and no recognizable scene. The flat frames failed the pixel check. Visual review identified the patterned frames, which had enough luminance variation to pass that check. The ARM CPU control rendered a recognizable apple scene with the same Tiny weights and TAESD, so the valid on-device comparison uses the CPU. [Android setup](../android/README.md)
