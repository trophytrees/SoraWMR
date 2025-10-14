# Fine-tuning Workflow Helpers

These scripts let you iterate on the detector without touching the production setup.

## 1. Back up the current dataset and weights

```powershell
conda activate sorawmr
python tools/backup_assets.py
```

Backups go to `backups/sorawm_backup_<timestamp>/` and include a copy of `datasets/` plus `resources/best.pt`. Pass `--dest` if you want a custom folder.

## 2. Export missed / low-confidence frames

```powershell
python tools/export_missed_frames.py video.mov another_clip.mp4 --threshold 0.45 --metadata datasets/missed_frames/export.csv
```

Every frame where the detector returns nothing, or confidence < 0.45, is written to `datasets/missed_frames/` (configurable with `--output`). The optional CSV keeps track of where each image came from.

Tips:

- Add more videos or use a text file list: `python tools/export_missed_frames.py --stride 2 @videos.txt`
- Lower `--stride` to scan every frame; raise it to speed up the pass on long videos.

## 3. Label and merge

Label the images in `datasets/missed_frames/` (YOLO format, class `0`). Drop the labeled image/label pairs into your training split—either extend `datasets/coco8/` or create a new dataset and point `train/train.py` to its YAML.

## 4. Fine-tune

Once the new labels are in place, run the existing training entrypoint:

```powershell
python train/train.py
```

Replace `resources/best.pt` with the new weights when you are satisfied (keeping the backup from step 1).

## 5. Validate

Use the visualizer wrapper to sanity-check the troublesome clips:

```powershell
.\run_visualizer_custom.ps1 -VideoPath .\video.mov
```

Repeat the cycle if you still see misses—each pass will add targeted samples and make the detector more robust.
