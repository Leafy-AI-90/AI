"""Read local author archives without duplicating every image on disk."""
import csv
import io
from pathlib import Path, PurePosixPath
import zipfile
from PIL import Image
from torch.utils.data import Dataset


class ArchiveDataset(Dataset):
    def __init__(self, root, name, split, transform=None):
        if name not in ('plantvillage', 'plantdoc') or split not in ('train', 'validation', 'test'):
            raise ValueError('Choose a supported dataset and train/validation/test split')
        self.root = Path(root).resolve()
        self.transform = transform
        with (self.root / f'data/processed/{name}.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        self.classes = sorted({r['label'] for r in rows if r['split'] != 'quarantine'})
        self.class_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.rows = [r for r in rows if r['split'] == split]
        if not self.rows:
            raise ValueError(f'No {split} images in {name}')
        for row in self.rows:
            archive = (self.root / row['archive']).resolve()
            if not archive.is_relative_to(self.root):
                raise ValueError('Archive path outside project')
        self._archives = {}

    def __getstate__(self):
        # Windows DataLoader workers receive their own archive handles.
        state = self.__dict__.copy()
        state['_archives'] = {}
        return state

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        archive = self._archives.setdefault(row['archive'], None)
        if archive is None:
            archive = zipfile.ZipFile(self.root / row['archive'])
            self._archives[row['archive']] = archive
        with Image.open(io.BytesIO(archive.read(row['member']))) as image:
            image = image.convert('RGB')
        return (self.transform(image) if self.transform else image,
                self.class_to_idx[row['label']])

    def close(self):
        for archive in self._archives.values():
            if archive is not None:
                archive.close()
        self._archives.clear()
