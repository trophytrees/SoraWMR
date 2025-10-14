from pathlib import Path
from typing import Callable

import cv2
import ffmpeg
import numpy as np
from loguru import logger
from tqdm import tqdm

from sorawm.utils.video_utils import VideoLoader
from sorawm.watermark_cleaner import WaterMarkCleaner
from sorawm.watermark_detector import SoraWaterMarkDetector


class SoraWM:
    def __init__(self):
        self.detector = SoraWaterMarkDetector()
        self.cleaner = WaterMarkCleaner()

    def run(
        self,
        input_video_path: Path,
        output_video_path: Path,
        progress_callback: Callable[[int], None] | None = None,
    ):
        input_video_loader = VideoLoader(input_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        width = input_video_loader.width
        height = input_video_loader.height
        fps = input_video_loader.fps
        total_frames = input_video_loader.total_frames
        

        
        temp_output_path = output_video_path.parent / f"temp_{output_video_path.name}"
        output_options = {
            "pix_fmt": "yuv420p",
            "vcodec": "libx264",
            "preset": "slow",  
        }
        output_options["r"] = fps
        
        if input_video_loader.original_bitrate:
            output_options["video_bitrate"] = str(int(int(input_video_loader.original_bitrate) * 1.2))
        else:
            output_options["crf"] = "18"
        
        process_out = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s=f"{width}x{height}",
                r=fps,
            )
            .output(str(temp_output_path), **output_options)
            .overwrite_output()
            .global_args("-loglevel", "error")
            .run_async(pipe_stdin=True)
        )

        frame_and_mask = {}
        detect_missed = []

        logger.debug(
            f"total frames: {total_frames}, fps: {fps}, width: {width}, height: {height}"
        )
        for idx, frame in enumerate(
            tqdm(input_video_loader, total=total_frames, desc="Detect watermarks")
        ):
            detection_result = self.detector.detect(frame)
            if detection_result["detected"] and detection_result.get("detections"):
                frame_and_mask[idx] = {
                    "frame": frame,
                    "detections": detection_result["detections"],
                }
            else:
                frame_and_mask[idx] = {"frame": frame, "detections": None}
                detect_missed.append(idx)

            # 10% - 50%
            if progress_callback and idx % 10 == 0:
                progress = 10 + int((idx / total_frames) * 40)
                progress_callback(progress)

        logger.debug(f"detect missed frames: {detect_missed}")

        for missed_idx in detect_missed:
            before = max(missed_idx - 1, 0)
            after = min(missed_idx + 1, total_frames - 1)
            before_det = frame_and_mask[before]["detections"]
            after_det = frame_and_mask[after]["detections"]
            if before_det:
                frame_and_mask[missed_idx]["detections"] = [dict(d) for d in before_det]
            elif after_det:
                frame_and_mask[missed_idx]["detections"] = [dict(d) for d in after_det]

        for idx in tqdm(range(total_frames), desc="Remove watermarks"):
            frame_info = frame_and_mask[idx]
            frame = frame_info["frame"]
            detections = frame_info["detections"]
            if detections:
                mask = np.zeros((height, width), dtype=np.uint8)
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    cls_id = det.get("class", 0)
                    x1_clamped = max(0, min(width, x1))
                    x2_clamped = max(0, min(width, x2))
                    y1_clamped = max(0, min(height, y1))
                    y2_clamped = max(0, min(height, y2))
                    if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
                        continue
                    w = x2_clamped - x1_clamped
                    h = y2_clamped - y1_clamped
                    if cls_id == 1:  # text region
                        pad_left = max(4, int(w * 0.12))
                        pad_right = max(5, int(w * 0.14))
                        pad_y = max(4, int(h * 0.16))
                    else:  # icon or fallback
                        pad_left = max(6, int(w * 0.18))
                        pad_right = max(6, int(w * 0.18))
                        pad_y = max(6, int(h * 0.2))
                    x1_p = max(0, x1_clamped - pad_left)
                    x2_p = min(width, x2_clamped + pad_right)
                    y1_p = max(0, y1_clamped - pad_y)
                    y2_p = min(height, y2_clamped + pad_y)
                    mask[y1_p:y2_p, x1_p:x2_p] = 255

                kernel_width = max(5, int(width * 0.008)) | 1
                kernel_height = max(5, int(height * 0.01)) | 1
                dilation_kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (kernel_width, kernel_height)
                )
                mask = cv2.dilate(mask, dilation_kernel, iterations=1)

                erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                inpaint_mask = cv2.erode(mask, erosion_kernel, iterations=1)
                if not inpaint_mask.any():
                    inpaint_mask = mask

                cleaned = self.cleaner.clean(frame, inpaint_mask)

                mask_float = mask.astype(np.float32) / 255.0
                feather = cv2.GaussianBlur(
                    mask_float,
                    (0, 0),
                    sigmaX=max(1.2, width * 0.0035),
                    sigmaY=max(1.2, height * 0.0035),
                )
                feather = np.clip(feather[..., None], 0.0, 1.0)

                cleaned_frame = (
                    feather * cleaned.astype(np.float32)
                    + (1.0 - feather) * frame.astype(np.float32)
                ).astype(np.uint8)
            else:
                cleaned_frame = frame
            process_out.stdin.write(cleaned_frame.tobytes())

            # 50% - 95%
            if progress_callback and idx % 10 == 0:
                progress = 50 + int((idx / total_frames) * 45)
                progress_callback(progress)

        process_out.stdin.close()
        process_out.wait()

        # 95% - 99%
        if progress_callback:
            progress_callback(95)

        self.merge_audio_track(input_video_path, temp_output_path, output_video_path)

        if progress_callback:
            progress_callback(99)

    def merge_audio_track(
        self, input_video_path: Path, temp_output_path: Path, output_video_path: Path
    ):
        logger.info("Merging audio track...")
        video_stream = ffmpeg.input(str(temp_output_path))
        audio_stream = ffmpeg.input(str(input_video_path)).audio

        (
            ffmpeg.output(
                video_stream,
                audio_stream,
                str(output_video_path),
                vcodec="copy",
                acodec="aac",
            )
            .overwrite_output()
            .run(quiet=True)
        )
        # Clean up temporary file
        temp_output_path.unlink()
        logger.info(f"Saved no watermark video with audio at: {output_video_path}")


if __name__ == "__main__":
    from pathlib import Path

    input_video_path = Path(
        "resources/19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4"
    )
    output_video_path = Path("outputs/sora_watermark_removed.mp4")
    sora_wm = SoraWM()
    sora_wm.run(input_video_path, output_video_path)
