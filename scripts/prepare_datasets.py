"""Build local CSV manifests; preserve test groups and split training groups only."""
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
import re
import random
import zipfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def leaf_keys(member, label, leaf_map):
    stem = PurePosixPath(member).name.replace('_final_masked', '').split('___')[-1]
    stem = stem.split('copy')[0]
    stem = re.sub(r'\.(jpg|jpeg|png)$', '', stem, flags=re.I).strip().lower()
    matches = [value for value in leaf_map.get(stem, []) if value.split(':::')[0] == label]
    if matches:
        return ['leaf:' + value for value in matches], 'upstream_leaf_map'
    return ['fallback:' + label + ':' + stem], 'filename_fallback_not_verified_plant_id'


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def rows_from_archive(name):
    directory = ROOT / 'data/raw' / name
    rows, grouping = [], Groups()
    leaf_map, official = {}, {}
    if name == 'plantvillage':
        leaf_map = json.loads((directory / 'leaf-map.json').read_text(encoding='utf-8'))
        for split in ('train', 'test'):
            for line in (directory / f'upstream_{split}.txt').read_text(encoding='utf-8').splitlines():
                if line.strip() in official:
                    raise ValueError('Duplicate membership in upstream split files')
                official[line.strip()] = split
    with zipfile.ZipFile(directory / 'images.zip') as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or path.suffix.lower() not in EXTENSIONS or '__MACOSX' in path.parts:
                continue
            parts = path.parts
            if name == 'plantvillage':
                if len(parts) < 4 or parts[:2] != ('raw', 'color'):
                    continue
                label = parts[2]
                upstream = official.get(info.filename)
                if upstream is None:
                    raise ValueError(f'Image missing upstream split: {info.filename}')
                keys, source = leaf_keys(info.filename, label, leaf_map)
            else:
                if len(parts) < 4 or parts[1] not in ('train', 'test'):
                    continue
                upstream, label = parts[1], parts[2]
                # Crops of a source photo may share basename. Preserve these together
                # across classes too; label conflicts will be quarantined below.
                keys = ['filename:' + path.name.lower()]
                source = 'filename_and_exact_bytes_only_no_plant_metadata'
            payload = archive.read(info)
            payload_hash = hashlib.sha256(payload).hexdigest()
            image_valid = 'true'
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
            except (OSError, ValueError, SyntaxError):
                image_valid = 'false'
            for key in keys:
                grouping.union('bytes:' + payload_hash, key)
            rows.append(dict(dataset=name, archive=f'data/raw/{name}/images.zip',
                member=info.filename, label=label, upstream_split=upstream,
                group_source=source, sha256=payload_hash, image_valid=image_valid))
    if not rows:
        raise ValueError(f'No images found for {name}; archive structure changed')
    if official and set(official) != {row['member'] for row in rows}:
        raise ValueError('Upstream split file and archive contents differ')
    for row in rows:
        row['group_id'] = hashlib.sha256(grouping.find('bytes:' + row['sha256']).encode()).hexdigest()[:24]
    return rows


def assign_splits(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row)
    eligible = defaultdict(list)
    for key, members in groups.items():
        if len({m['label'] for m in members}) > 1 or any(m['image_valid'] == 'false' for m in members):
            split = 'quarantine'
        elif any(m['upstream_split'] == 'test' for m in members):
            split = 'test'
        else:
            split = 'train'
            eligible[members[0]['label']].append(key)
        for member in members:
            member['split'] = split
    rng = random.Random(42)
    for label, keys in sorted(eligible.items()):
        keys.sort()
        rng.shuffle(keys)
        n = min(len(keys) - 1, max(1, round(len(keys) * 0.2))) if len(keys) > 1 else 0
        for key in keys[:n]:
            for member in groups[key]:
                member['split'] = 'validation'
    return rows


def main():
    all_rows = {name: assign_splits(rows_from_archive(name)) for name in ('plantvillage', 'plantdoc')}
    # Prevent any exact PlantDoc test image appearing in PlantVillage training.
    pd_test_hashes = {r['sha256'] for r in all_rows['plantdoc'] if r['split'] == 'test'}
    affected = {r['group_id'] for r in all_rows['plantvillage']
                if r['sha256'] in pd_test_hashes and r['split'] in ('train', 'validation')}
    for row in all_rows['plantvillage']:
        if row['group_id'] in affected:
            row['split'] = 'quarantine'
    output = ROOT / 'data/processed'
    output.mkdir(parents=True, exist_ok=True)
    summary = {'seed': 42, 'validation_fraction_of_upstream_train_groups': 0.2,
               'cross_dataset_test_overlap_groups_quarantined': len(affected), 'datasets': {}}
    for name, rows in all_rows.items():
        with (output / f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        sets = defaultdict(set)
        for row in rows:
            sets[row['split']].add(row['group_id'])
        assert not sets['train'] & (sets['validation'] | sets['test'])
        assert not sets['validation'] & sets['test']
        summary['datasets'][name] = {
            'images': len(rows), 'classes': sorted({r['label'] for r in rows}),
            'split_counts': dict(Counter(r['split'] for r in rows)),
            'unreadable_images': sum(r['image_valid'] == 'false' for r in rows),
            'group_sources': dict(Counter(r['group_source'] for r in rows)),
            'upstream_train_moved_to_test': sum(r['upstream_split'] == 'train' and r['split'] == 'test' for r in rows),
            'archive_sha256': json.loads((ROOT / f'data/raw/{name}/download-receipt.json').read_text())['sha256']}
    (ROOT / 'data/dataset-summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
