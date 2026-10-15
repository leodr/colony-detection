# Data

The images are not stored in git. They are in the private Hugging Face dataset
[leodriesch/photobox-colonies](https://huggingface.co/datasets/leodriesch/photobox-colonies).
Restore this folder with:

```bash
python scripts/download_data.py
```

| Folder | Content |
|---|---|
| `labeled/images` | 251 counted photos (3040 × 3040 JPEG) |
| `labeled/masks` | 200 instance segmentation masks |
| `labeled/circle_masks` | 251 plate masks (agar area) |
| `labeled/pen_annotated` | 196 photos with colonies circled by hand, `<name>_<count>.jpg` |
| `unlabeled` | 278 photos without labels |
| `labeled/metadata.csv` | manual counting sheet (in git) |
| `labeled/splits.json` | train/val/test split (in git) |

The AGAR dataset is public at [leodriesch/agar](https://huggingface.co/datasets/leodriesch/agar).
