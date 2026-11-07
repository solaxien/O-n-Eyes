"""Assemble inspectable browser evidence without re-running any generation."""
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from p3.provenance import file_hash, write_json

root = Path('.runs/browser')
destination = Path('docs/browser')
destination.mkdir(parents=True, exist_ok=True)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def experiment(name, score=None):
    directory = root/name
    protocol = read(directory/'experiment.json')
    records = [read(directory/f"{sample['id']}.json") for sample in protocol['manifest']]
    times = [r['metadata']['submit_to_visible_ms']/1000 for r in records]
    memory = read(directory/'memory.json')
    result = {
        'protocol': protocol, 'records': records,
        'completion': read(directory/'completion.json'),
        'submit_to_visible_s': {'n': len(times), 'min': min(times), 'p50': float(np.median(times)),
                                'p95': float(np.percentile(times, 95)), 'max': max(times)},
        'memory': {key: value for key, value in memory.items() if key != 'samples'},
        'memory_file_sha256': file_hash(directory/'memory.json'),
        'warmup': read(directory/'warmup.json') if (directory/'warmup.json').exists() else None,
    }
    if score:
        result['scores'] = read(root/'scored'/score/'scores.json')
    return result


report = {'schema': 1, 'date': '2026-11-07',
          'scope': 'Poco F6 browser inference; no T4/external GPU. Laptop RTX 4050 used only for export and offline scoring.',
          'model': read(Path('browser/model-manifest.json')),
          'offline': experiment('offline-acceptance-v1'),
          'steps': {str(n): experiment(f'composition-{n}-v1', f'steps{n}-full') for n in [4, 6, 12]},
          'cv': experiment('cv-6-v1'),
          'cv_scores': {kind: read(root/'scored'/f'cv6-{kind}'/'scores.json')
                        for kind in ['raw', 'enhanced', 'omit-repair', 'omit-edges', 'omit-color', 'omit-gate']},
          'quality384': experiment('quality384-v1'),
          'desktop': read(root/'desktop-check.json'),
          'setup': read(root/'public-setup-recovered.json')}
report['offline']['isolation'] = read(root/'offline-acceptance-v1/offline-state.json')
report['offline']['restored'] = read(root/'offline-acceptance-v1/network-restored.json')
report['offline']['network_events'] = [json.loads(line) for line in (root/'offline-acceptance-v1/network.jsonl').read_text().splitlines()]
if (root/'v3-component-probes.json').exists():
    report['numerical_probes'] = read(root/'v3-component-probes.json')
if (root/'published-offline-v1/completion.json').exists():
    report['published_offline'] = experiment('published-offline-v1')
    report['published_offline']['isolation'] = read(root/'published-offline-v1/offline-state.json')
    report['published_offline']['network_events'] = [json.loads(line) for line in (root/'published-offline-v1/network.jsonl').read_text().splitlines()]
if (root/'published-visible-v1/completion.json').exists():
    report['published_visible'] = experiment('published-visible-v1')
    report['published_visible']['isolation'] = read(root/'published-visible-v1/offline-state.json')
    report['published_visible']['restored'] = read(root/'published-visible-v1/network-restored.json')
    report['published_visible']['network_events'] = [json.loads(line) for line in (root/'published-visible-v1/network.jsonl').read_text().splitlines()]
report['public_assets'] = read(root/'published-assets.json')
report['release_integrity'] = read(root/'release-integrity.json')
report['cache_transfer'] = read(root/'published-cache-transfer.json')
if (root/'cleanup.json').exists():
    report['cleanup'] = read(root/'cleanup.json')
write_json(destination/'evidence.json', report)

# Contact sheets are evaluation figures. Original PNGs remain unmodified.
def font_for(size, bold=False):
    for candidate in [('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'),
                      ('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf')]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


font = font_for(17)
title = font_for(22, bold=True)
for steps in [4, 6, 12]:
    canvas = Image.new('RGB', (800, 956), '#101717')
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 14), f'{steps} steps · Poco F6 · native 256 px · all 16 prompts', font=title, fill='#e8eeee')
    rows = report['steps'][str(steps)]['protocol']['manifest']
    for i, row in enumerate(rows):
        x, y = (i % 4)*200+10, (i//4)*224+56
        with Image.open(root/f'composition-{steps}-v1'/f"{row['id']}-enhanced.png") as image:
            canvas.paste(image.resize((180, 180), Image.Resampling.LANCZOS).convert('RGB'), (x, y))
        draw.text((x, y+186), row['id'], font=font, fill='#b4f1d9')
    canvas.save(destination/f'steps-{steps}.jpg', quality=91)
shutil.copyfile(root/'offline-acceptance-v1/offline-fresh-627194-enhanced.png', destination/'offline-result.png')
shutil.copyfile(root/'quality384-v1/red-apple-left-enhanced.png', destination/'quality384-missing-cup.png')

print(json.dumps({'evidence': str(destination/'evidence.json'), 'steps': {
    key: {'seconds': value['submit_to_visible_s'], 'clip': value['scores']['clip_mean'],
          'peak_pss_mib': value['memory']['peak_pss_mib']}
    for key, value in report['steps'].items()}}, indent=2))
