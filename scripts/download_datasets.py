"""Download pinned author-hosted archives; never execute upstream dataset code."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        checksum = digest(path)
        if expected and checksum != expected:
            raise ValueError(f'Checksum mismatch in existing file: {path}')
        return checksum
    temporary = path.with_suffix(path.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'Leafy-AI-dataset-setup/0.1'})
    h, received, last_report = hashlib.sha256(), 0, 0
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as f:
        while block := response.read(4 * 1024 * 1024):
            f.write(block)
            h.update(block)
            received += len(block)
            if received - last_report >= 256 * 1024 * 1024:
                print(f'{path.parent.name}/{path.name}: {received / 1024**2:.0f} MiB', flush=True)
                last_report = received
    checksum = h.hexdigest()
    if expected and checksum != expected:
        raise ValueError(f'Download checksum mismatch: {path}')
    temporary.replace(path)
    return checksum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['all', 'plantvillage', 'plantdoc'], default='all')
    args = parser.parse_args()
    sources = json.loads((ROOT / 'data/sources.json').read_text(encoding='utf-8'))
    for name, source in sources.items():
        if args.dataset not in ('all', name):
            continue
        target = ROOT / 'data/raw' / name
        checksum = download(source['url'], target / 'images.zip', source.get('sha256'))
        if source.get('bytes') and (target / 'images.zip').stat().st_size != source['bytes']:
            raise ValueError('Unexpected archive size')
        with zipfile.ZipFile(target / 'images.zip') as archive:
            count = len(archive.infolist())
        if name == 'plantvillage':
            base = f"{source['download_repository']}/resolve/{source['revision']}"
            for remote, local in [('leaf_grouping/leaf-map.json', 'leaf-map.json'),
                                  ('splits/color_train.txt', 'upstream_train.txt'),
                                  ('splits/color_test.txt', 'upstream_test.txt'),
                                  ('README.md', 'SOURCE_README.md')]:
                download(base + '/' + remote, target / local)
        else:
            base = f"https://raw.githubusercontent.com/pratikkayal/PlantDoc-Dataset/{source['revision']}"
            download(base + '/LICENSE.txt', target / 'LICENSE.txt')
            download(base + '/README.md', target / 'SOURCE_README.md')
        receipt = dict(source, sha256=checksum, archive_entries=count,
                       downloaded_bytes=(target / 'images.zip').stat().st_size)
        (target / 'download-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        print(f'{name}: verified archive, {count} entries', flush=True)


if __name__ == '__main__':
    main()
