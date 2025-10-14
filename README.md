# SoraWatermarkCleaner

English | [中文](README-zh.md)

This project provides an elegant way to remove the sora watermark in the sora2 generated videos.


- Watermark removed

https://github.com/user-attachments/assets/8cdc075e-7d15-4d04-8fa2-53dd287e5f4c

- Original

https://github.com/user-attachments/assets/3c850ff1-b8e3-41af-a46f-2c734406e77d

⭐️: **Yolo weights has been updated, try the new version watermark detect model, it should work better.**

## Features

- **High fidelity removal pipeline**: YOLOv11-based detector with automatic weight downloads, device selection, missed-frame backfilling, adaptive padding, LaMa inpainting, and feathered alpha blending while preserving original audio via ffmpeg remux.
- **Turnkey Python + CLI tooling**: Simple `SoraWM().run` API with progress callbacks, `example.py` usage, detection preview generation, visualization script, and helpers for metadata, thumbnails, and per-frame exports.
- **Interactive Streamlit app**: Drag-and-drop uploads, live progress telemetry, staged status messaging, inline playback, and one-click downloads of cleaned videos.
- **Production FastAPI service**: Queue-backed worker with SQLite tracking, background thumbnails, `/submit_remove_task` → `/get_results` → `/download` endpoints, automatic model reloads, and OpenAPI docs.
- **Integrated web workspace**: Jinja/Tailwind dashboard for video library management, job monitoring, detection previews, manual annotation studio, Roboflow auto-labeling, YOLO fine-tuning launches, dataset summaries, and output browsing.
- **Model and dataset ops**: Weight activation/rename/delete with backups, Cog/Replicate packaging, fine-tuning scripts, `tools/backup_assets.py`, `tools/export_missed_frames.py`, and Roboflow automation CLI for continuous improvement.

## 1. Method

The SoraWatermarkCleaner(we call it `SoraWm` later) is composed of two parsts:

- SoraWaterMarkDetector: We trained a yolov11s version to detect the sora watermark. (Thank you yolo!)

- WaterMarkCleaner: We refer iopaint's implementation for watermark removal using the lama model.

  (This codebase is from https://github.com/Sanster/IOPaint#, thanks for their amazing work!)

Our SoraWm is purely deeplearning driven and yields good results in many generated videos.



## 2. Installation

[FFmpeg](https://ffmpeg.org/) is needed for video processing, please install it first.  We highly recommend using the `uv` to install the environments:

1. installation:

```bash
uv sync
```

> now the envs will be installed at the `.ven`, you can activate the env using:
>
> ```bash
> source .venv/bin/activate
> ```

2. Downloaded the pretrained models:

The trained yolo weights will be stored in the `resources` dir as the `best.pt`.  And it will be automatically download from https://github.com/linkedlist771/SoraWatermarkCleaner/releases/download/V0.0.1/best.pt . The `Lama` model is downloaded from https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt, and will be stored in the torch cache dir. Both downloads are automatic, if you fail, please check your internet status.

## 3.  Demo

To have a basic usage, just try the `example.py`:

```python

from pathlib import Path
from sorawm.core import SoraWM


if __name__ == "__main__":
    input_video_path = Path(
        "resources/dog_vs_sam.mp4"
    )
    output_video_path = Path("outputs/sora_watermark_removed.mp4")
    sora_wm = SoraWM()
    sora_wm.run(input_video_path, output_video_path)

```

We also provide you with a `streamlit` based interactive web page, try it with:

```bash
streamlit run app.py
```

<img src="resources/app.png" style="zoom: 25%;" />

## **4. WebServer**

Here, we provide a **FastAPI-based web server** that can quickly turn this watermark remover into a service.

Simply run:

```
python start_server.py
```

The web server will start on port **5344**.

You can view the FastAPI [documentation](http://localhost:5344/docs) for more details.

There are three routes available:

1. **submit_remove_task**

   > After uploading a video, a task ID will be returned, and the video will begin processing immediately.

<img src="resources/53abf3fd-11a9-4dd7-a348-34920775f8ad.png" alt="image" style="zoom: 25%;" />

2. **get_results**

You can use the task ID obtained above to check the task status.

It will display the percentage of video processing completed.

Once finished, the returned data will include a **download URL**.

3. **download**

You can use the **download URL** from step 2 to retrieve the cleaned video.

### Inline Annotation & Fine-tuning

The dashboard also ships with an **Annotation Studio** so you can keep improving the detector without leaving the app:
- Pull any uploaded video into the studio, scrub to a tricky frame, and draw boxes over the missed watermark pieces.
- Saving writes the frame and YOLO labels to `datasets/manual_annotations/` and bumps the dataset counter in the UI.
- Point the built-in finetuning form at `datasets/manual_annotations/data.yaml` to retrain on the new samples.

Each manual save updates the dataset summary so it's easy to judge when you have enough examples for a new training run.


## 5. API

Packaged as a Cog and [published to Replicate](https://replicate.com/uglyrobot/sora2-watermark-remover) for simple API based usage.

## 6. License

 Apache License


## 7. Citation

If you use this project, please cite:

```bibtex
@misc{sorawatermarkcleaner2025,
  author = {linkedlist771},
  title = {SoraWatermarkCleaner},
  year = {2025},
  url = {https://github.com/linkedlist771/SoraWatermarkCleaner}
}
```

## 8. Acknowledgments

- [IOPaint](https://github.com/Sanster/IOPaint) for the LAMA implementation
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection

## 9. Fine-tuning Helpers

Need to harvest missed detections or back up assets before retraining? See `docs/fine_tune_workflow.md` for scripts that automate the loop without altering the current app.
