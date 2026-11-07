"""Import device-captured PNGs into the existing strict semantic scorer.

Generation is never performed here. Original browser PNGs/records remain
unchanged; RGB conversion is only for the scorer's image-file contract.
"""
import argparse
import json
from pathlib import Path

from PIL import Image
from p3.provenance import (establish_protocol, exclusive_run, file_hash,
                           save_image, source_hashes, write_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--kind', choices=['raw', 'enhanced'], default='enhanced')
    args = parser.parse_args()
    experiment = json.loads((args.source/'experiment.json').read_text())
    complete = json.loads((args.source/'completion.json').read_text())
    if complete['complete'] != len(experiment['manifest']):
        raise ValueError('Browser experiment is incomplete')
    if args.out.exists():
        raise ValueError('Use a new output directory; existing evidence is preserved.')
    records = []
    for sample in experiment['manifest']:
        record_path = args.source/f"{sample['id']}.json"
        record = json.loads(record_path.read_text())
        if record['sample'] != sample or record.get('warmup'):
            raise ValueError('Captured sample identity mismatch')
        artifact = record['artifacts'][args.kind]
        image = args.source/artifact['file']
        if file_hash(image) != artifact['sha256']:
            raise ValueError('Captured browser image has changed bytes')
        records.append((sample, record, image, file_hash(record_path)))
    with exclusive_run(args.out):
        fingerprint = establish_protocol(args.out, {
            'schema': 1, 'execution': 'Poco F6 visitor browser; imported for scoring only',
            'source': source_hashes('import_browser_run.py'), 'kind': args.kind,
            'experiment': experiment, 'experiment_sha256': file_hash(args.source/'experiment.json'),
            'completion_sha256': file_hash(args.source/'completion.json'),
            'captured_records': {s['id']: sha for s, _, _, sha in records},
            'timing_scope': 'Browser submit-to-visible, including loading if warm_models is false, CV and resize.',
        })
        for sample, captured, image, _ in records:
            with Image.open(image) as decoded:
                artifact = save_image(args.out/'images'/f"{sample['id']}.png", decoded)
            meta = captured['metadata']
            write_json(args.out/'runs'/f"{sample['id']}.json", {
                'sample': sample, 'status': 'complete', 'protocol_hash': fingerprint,
                'artifacts': {'images': artifact},
                'generation_ms': meta['submit_to_visible_ms'],
                'cv_ms': meta['stages_ms']['cv_resize_encode'],
                'cv_included_in_generation': True, 'browser_metadata': meta,
            })
    print(json.dumps({'out': str(args.out), 'samples': len(records), 'kind': args.kind}))


if __name__ == '__main__':
    main()
