from pathlib import Path
from typing import Tuple

import base64
import cv2
import ffmpeg
import numpy as np


class VideoLoader:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.get_video_info()

    def get_video_info(self):
        probe = ffmpeg.probe(self.video_path)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        width = int(video_info["width"])
        height = int(video_info["height"])
        fps = eval(video_info["r_frame_rate"])
        self.width = width
        self.height = height
        self.fps = fps
        if "nb_frames" in video_info:
            self.total_frames = int(video_info["nb_frames"])
        else:
            # 通过时长计算
            duration = float(video_info.get("duration", probe["format"]["duration"]))
            self.total_frames = int(duration * self.fps)

        original_bitrate = video_info.get("bit_rate", None)
        self.original_bitrate = original_bitrate

    def __len__(self):
        return self.total_frames

    def __iter__(self):
        process_in = (
            ffmpeg.input(self.video_path)
            .output("pipe:", format="rawvideo", pix_fmt="bgr24")
            .global_args("-loglevel", "error")
            .run_async(pipe_stdout=True)
        )

        try:
            while True:
                in_bytes = process_in.stdout.read(self.width * self.height * 3)
                if not in_bytes:
                    break

                frame = np.frombuffer(in_bytes, np.uint8).reshape(
                    [self.height, self.width, 3]
                )
                yield frame
        finally:
            # 确保进程被清理
            process_in.stdout.close()
            if process_in.stderr:
                process_in.stderr.close()
            process_in.wait()


def capture_frame(video_path: Path, timestamp: float) -> Tuple[np.ndarray, float]:
    """
    Return the frame at ``timestamp`` seconds and the actual timestamp captured.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    try:
        target_ms = max(timestamp, 0.0) * 1000.0
        cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        success, frame = cap.read()
        if not success or frame is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fallback_index = int(max(timestamp * fps, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_index)
            success, frame = cap.read()
        if not success or frame is None:
            raise RuntimeError("Failed to capture frame at the requested timestamp.")
        actual_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return frame, actual_timestamp
    finally:
        cap.release()


def frame_to_base64(frame: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def save_frame_image(video_path: Path, destination: Path, timestamp: float) -> Tuple[int, int, float]:
    """
    Save the frame at ``timestamp`` to ``destination``.
    Returns (width, height, actual_timestamp).
    """
    frame, actual_timestamp = capture_frame(video_path, timestamp)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), frame)
    height, width = frame.shape[:2]
    return width, height, actual_timestamp


def get_video_metadata(video_path: Path) -> dict:
    loader = VideoLoader(video_path)
    duration = loader.total_frames / loader.fps if loader.fps else 0
    return {
        "width": loader.width,
        "height": loader.height,
        "fps": loader.fps,
        "total_frames": loader.total_frames,
        "duration": duration,
    }


def generate_thumbnail(
    video_path: Path,
    destination: Path,
    *,
    timestamp: float = 1.0,
    width: int = 360,
) -> None:
    """
    Save a single-frame thumbnail for the given video.

    Args:
        video_path: Source video file.
        destination: Target image path.
        timestamp: Seconds into the video to capture.
        width: Output width in pixels, preserves aspect ratio.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        (
            ffmpeg.input(str(video_path), ss=max(timestamp, 0))
            .filter("scale", width, -1)
            .output(str(destination), vframes=1)
            .overwrite_output()
            .global_args("-loglevel", "error")
            .run()
        )
    except ffmpeg.Error as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to generate thumbnail: {exc}") from exc


if __name__ == "__main__":
    from tqdm import tqdm

    video_path = Path("resources/dog_vs_sam.mp4")

    # 创建 VideoLoader 实例
    loader = VideoLoader(video_path)

    # 显示视频信息
    print(f"视频路径: {video_path}")
    print(f"分辨率: {loader.width}x{loader.height}")
    print(f"帧率: {loader.fps:.2f} fps")
    print(f"总帧数: {loader.total_frames}")
    print(f"时长: {loader.total_frames / loader.fps:.2f} 秒")
    print("-" * 50)

    # 遍历所有帧并显示进度
    frame_count = 0
    for frame in tqdm(loader, total=len(loader), desc="处理视频"):
        frame_count += 1

        # 每隔 30 帧显示一次信息（可选）
        if frame_count % 30 == 0:
            print(
                f"\n第 {frame_count} 帧 - shape: {frame.shape}, dtype: {frame.dtype}, "
                f"min: {frame.min()}, max: {frame.max()}"
            )

    print(f"\n处理完成！共处理 {frame_count} 帧")

    # 测试提前退出（验证资源清理）
    print("\n测试提前退出...")
    loader2 = VideoLoader(video_path)
    for i, frame in enumerate(loader2):
        if i >= 5:  # 只读取前 5 帧
            print(f"提前退出，已读取 {i+1} 帧")
            break
    print("资源已正确清理")
