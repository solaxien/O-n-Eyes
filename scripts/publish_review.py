"""Render the measured review snapshot as a responsive static app page."""
from html import escape
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root / 'docs/review-evidence.json').read_text(encoding='utf-8'))


def table(rows, cv=False):
    heads = ['Configuration', 'CLIP', 'CV ms' if cv else 'Median seconds', 'Exact prompts', 'Count', 'Color', 'Position']
    text = '<div class="scroll" tabindex="0"><table><thead><tr>' + ''.join('<th>' + s + '</th>' for s in heads) + '</tr></thead><tbody>'
    for row in rows:
        cost = row['cv_p50_ms'] if cv else row['latency_p50_ms'] / 1000
        values = [row['id'], f"{row['clip_mean']:.4f}", f'{cost:.3f}', f"{round(row['composition_pass'] * 16)}/16"]
        values += [f"{row['clauses'][key]['passed']}/{row['clauses'][key]['total']}" for key in ['count', 'color', 'position']]
        text += '<tr>' + ''.join('<td>' + escape(v) + '</td>' for v in values) + '</tr>'
    return text + '</tbody></table></div>'


content = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Implementation review · P3 Image Lab</title><link rel="stylesheet" href="style.css">
<style>main{padding-bottom:3rem}.panel{padding:1.4rem;margin:1.5rem 0}h2{margin-bottom:1rem}p{line-height:1.65}table{white-space:nowrap;width:100%;border-collapse:collapse}th,td{padding:.65rem;text-align:left;border-bottom:1px solid #344343}.scroll{overflow:auto}a{color:#a3eed6}</style></head><body><main class="shell">
<div class="intro"><p class="eyebrow">O(n) Eyes · measured review · 7 November 2026</p><h1>Implementation review</h1>
<p><a href="/">Back to Image Lab</a></p><p>This review keeps the original COCO and two-prompt phone studies intact. New results below use 16 paired prompts per configuration. No T4 or external GPU was used.</p></div>
<section class="panel"><h2>Reliability fixes</h2><p>Strict experiment fingerprints, image checksums, safe resumption, score invalidation, adapted-model export verification, training-cache checks, and recoverable adaptive generation. Twenty-nine targeted regression checks passed locally. Six public desktop/mobile-width requests displayed images in 0.76–2.28 seconds; Chrome on the physical Poco F6 took 2.42 seconds in a separate check. All used this laptop's GPU.</p></section>
<section class="panel"><h2>Local GPU · controlled pilot</h2><p>RTX 4050 Laptop GPU, native 512 px. Global CLIP and exact object/color/position checks measure different properties. The six-step row's one exact pass does not establish a general quality improvement.</p>'''
content += table(data['host']) + '</section><section class="panel"><h2>Classical vision · same decoded pixels</h2><p>CV reduced injected speckle error in controlled tests. It does not repair missing objects. Full CV lost one automatic count and color clause in this small set; Raw remains available.</p>'
content += table(data['cv'], True) + '</section><section class="panel"><h2>Poco F6 · native CPU</h2><p>Six steps, 256 px, 16 prompts each. Base and trained-adapter exports are verified separately. Resize, QA and optional CV run on the laptop. Different thermal conditions prevent attributing the latency difference to the adapter.</p>'
content += table(data['mobile']) + '''<p>Peak native RSS was 967 MiB for base and 1,514 MiB for adapted. Both scored 0/16 exact prompts. This is a benchmark executable, with no interactive Android APK.</p></section>
<section class="panel"><h2>Remaining limits</h2><p>Strong semantic consistency and ten-second native generation are unachieved. The public demo needs this laptop and its temporary tunnel to stay online. Permanent public hosting and Docker runtime verification remain pending. Failed Adreno outputs remain failures.</p>
<p><a href="https://github.com/solaxien/O-n-Eyes/blob/amg/p3-reliability-review/docs/REVIEW_RESULTS.md">Full results, methods and provenance</a> · <a href="https://github.com/solaxien/O-n-Eyes/blob/amg/p3-reliability-review/docs/JUDGING.md">Requirements and judging walkthrough</a></p></section></main></body></html>'''
(root / 'p3/web/review.html').write_text(content, encoding='utf-8')
