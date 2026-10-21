#!/usr/bin/env python3
"""Acquire a pinned, attributed human FLEURS subset; never fabricate audio."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = 'FluidInference/fleurs-full'
REVISION = '1cca811bb8ea4d370345f108f00518167040282c'


def fetch(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def acquire(root, languages, count):
    root.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 1, 'dataset': REPO, 'revision': REVISION,
                'license': 'CC-BY-4.0', 'source': 'https://huggingface.co/datasets/google/fleurs',
                'split': 'official-test-repackaged', 'seed': 'qualification-v1',
                'speaker_ids_available': False, 'purpose': 'monolingual ASR evaluation, not code-switch validation',
                'selection': 'lowest SHA256(seed/language/item) from pinned upstream test listing', 'samples': []}
    for lang in languages:
        # Pagination is mandatory: larger language splits exceed HF's page size.
        url = f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{lang}?limit=1000'
        listing = []
        while url:
            with urllib.request.urlopen(url, timeout=90) as response:
                listing.extend(json.load(response))
                links = response.headers.get('Link', '')
                url = next((part.split('>')[0].strip().lstrip('<') for part in links.split(',')
                            if 'rel="next"' in part), None)
        prefix = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'
        transcript = fetch(prefix + f'{lang}/{lang}.trans.txt').decode('utf-8')
        refs = {}
        for line in transcript.splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                if parts[0] in refs:
                    raise ValueError('duplicate transcript ID')
                refs[parts[0]] = parts[1]
        audio = [item for item in listing if item['path'].endswith(('.wav', '.flac'))]
        audio.sort(key=lambda item: sha(('qualification-v1/' + item['path']).encode()))
        selected = audio[:count]
        if len(selected) != count:
            raise ValueError(f'{lang}: requested {count}, only {len(selected)} available')
        directory = root / lang
        directory.mkdir(exist_ok=True)

        def download(item):
            name = Path(item['path']).name
            sample_id = Path(name).stem
            reference = refs[sample_id]
            target = directory / name
            data = target.read_bytes() if target.exists() else fetch(prefix + item['path'])
            digest = sha(data)
            expected = item.get('lfs', {}).get('oid')
            if expected and digest != expected:
                raise ValueError(f'upstream audio hash mismatch: {name}')
            if len(data) != item['size']:
                raise ValueError(f'audio length mismatch: {name}')
            if not target.exists():
                target.write_bytes(data)
            return {'language': lang, 'sample_id': sample_id, 'path': f'{lang}/{name}',
                    'sha256': digest, 'bytes': len(data), 'reference': reference,
                    'speaker_id': None, 'source_url': prefix + item['path']}

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(download, selected))
        (directory / f'{lang}.trans.txt').write_text(''.join(
            row['sample_id'] + ' ' + row['reference'] + '\n' for row in sorted(rows, key=lambda r: r['sample_id'])))
        manifest['samples'].extend(rows)
        print(f'{lang}: {len(rows)} audio hashes verified', flush=True)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    (root / 'ATTRIBUTION.md').write_text(
        '# Attribution\n\nFLEURS, Google et al., CC-BY-4.0. '
        'https://huggingface.co/datasets/google/fleurs\n\n'
        f'FluidInference test-set repackaging: {REPO}@{REVISION}.\n'
        'Changes: deterministic subset selection and local layout; audio unchanged. '
        'No new speaker or language-switch annotations are claimed.\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--languages', default='en_us,es_419,pt_br,fr_fr,de_de,cmn_hans_cn,ja_jp,ar_eg')
    parser.add_argument('--count', type=int, default=100)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('count must be positive')
    acquire(args.root, args.languages.split(','), args.count)
