# PlantVillage and PlantDoc in Leafy AI

## What is copied, and what belongs in Git?

Keep one local download of each dataset inside the project. Do not copy its
images into Git commits, fork the dataset into this code repository, or commit
another repository's `.git` folder. The archives are ordinary local data files.
They remain under `.gitignore`; only code, provenance and small summaries are
published to GitHub. Git LFS or DVC can be added later if you need to distribute
your own versioned dataset, but neither is needed for these upstream downloads.

```text
<repository root>/
  data/
    sources.json                 # tracked: pinned upstream revisions and provenance
    dataset-summary.json         # tracked: counts, audit and archive checksums
    raw/                         # ignored by Git
      plantvillage/
        images.zip               # author-hosted archive; loader uses color only
        leaf-map.json
        upstream_train.txt
        upstream_test.txt
        SOURCE_README.md
        download-receipt.json
      plantdoc/
        images.zip               # cropped classification dataset
        LICENSE.txt
        SOURCE_README.md
        download-receipt.json
    processed/                   # ignored by Git; reproducible split records
      plantvillage.csv
      plantdoc.csv
  models/weights/                # ignored pretrained checkpoints
  runs/                         # ignored future experiment outputs
```

The PyTorch loader reads images directly from the ZIP archives. There is no need
to unzip tens of thousands of files or maintain a second extracted copy. It does
not execute any upstream dataset loading scripts. The PlantVillage author archive
contains other image variants too; only color images enter our manifest.

To recreate the data from the pinned sources:

```bash
python scripts/download_datasets.py
python scripts/prepare_datasets.py
python scripts/check_dataset_batches.py
```

## How to train with them

Use PlantVillage for image-backbone pretraining, then optionally fine-tune on
PlantDoc's **training** portion for field-image adaptation. Validation is carved
from each training portion. Keep the test portions out of training and tuning.
If you train on PlantDoc, do not describe its training images as an unseen
external benchmark. To measure pure PlantVillage-to-PlantDoc transfer, freeze the
PlantVillage model first and evaluate on a reviewed overlapping label mapping.

```python
from torch.utils.data import DataLoader
from leafy_ai.datasets import ArchiveDataset
from leafy_ai.vision import image_transform, load_backbone

root = '.'  # run from repository root
train = ArchiveDataset(root, 'plantvillage', 'train', image_transform(training=True))
valid = ArchiveDataset(root, 'plantvillage', 'validation', image_transform())
loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=0)
model = load_backbone('efficientnet_b0', 'models/weights', num_classes=len(train.classes))
```

For PlantDoc adaptation, keep the learned feature extractor and replace/retrain
the final layer for `len(plantdoc.classes)`. Class indexes are dataset-specific;
do not concatenate these datasets or reuse classifier labels by numeric index.
The current archives have 38 PlantVillage labels and 28 PlantDoc folder labels.
No automatic disease-name equivalence or five-class Choy Sum mapping is assumed.

These are RGB disease datasets. They do not supply your paired 18-band spectra,
transmission measurements, or independently established Choy Sum N/Fe/Xcc labels.
They can train the image branch; they cannot by themselves train or validate
your spectral/fusion models or establish nutrient-deficiency performance.

## Split audit

Actual archive counts: 54,305 PlantVillage color images and 2,578 PlantDoc images.
These differ from headline paper counts; the installed archive and its checksums
are the reproducibility reference. Every selected image decoded successfully.

| Dataset | Train | Validation | Test | Quarantined |
|---|---:|---:|---:|---:|
| PlantVillage | 34,704 | 8,602 | 10,999 | 0 |
| PlantDoc | 1,855 | 461 | 226 | 36 |

Known leaf groups, duplicate file hashes and source filenames are kept together.
If a group touches the upstream test split, the group goes to test, never train.
This moved 290 originally training PlantVillage images and 3 PlantDoc images into
test to avoid known overlap. Validation uses a deterministic 20% of the remaining
training groups, stratified by label. Conflicting-label groups are quarantined;
PlantDoc same-basename grouping is conservative and may exclude unrelated photos
with generic identical filenames. Review the local CSV before relaxing it.

PlantVillage leaf metadata matched 40,490 images; 13,815 use filename fallbacks.
PlantDoc has no verified plant IDs. This is not proof of full plant independence:
near-duplicates, altered copies, and undocumented related crops require further
review. Exact PlantDoc test hashes were also checked against PlantVillage
train/validation; no additional overlap groups were found. Full cross-dataset
near-duplicate auditing is still required before combined-data experiments.

## Attribution and licensing

- PlantVillage: [author's GitHub repository](https://github.com/spMohanty/PlantVillage-Dataset)
  and [author-hosted download](https://huggingface.co/datasets/mohanty/PlantVillage).
  Cite Mohanty, Hughes and Salathe (2016), DOI `10.3389/fpls.2016.01419`.
  The author's dataset card labels it CC-BY-SA-3.0, but the GitHub repository has
  no standalone data license and its loader calls the license an assumption.
  Retain provenance and confirm terms before redistributing images.
- PlantDoc: [author's cropped classification repository](https://github.com/pratikkayal/PlantDoc-Dataset).
  It specifies CC-BY-4.0; the downloaded copy retains its license and README.
  Cite Singh et al. (2020), DOI `10.1145/3371158.3371196`.

The repository's MIT license covers our code, not these third-party datasets.
Only metadata and setup code are published; dataset images stay local.
