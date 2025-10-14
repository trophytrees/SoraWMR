from pathlib import Path

import cv2
from tqdm import tqdm

from sorawm.configs import ROOT

videos_dir = ROOT / "videos"
datasets_dir = ROOT / "datasets"
images_dir = datasets_dir / "images"
images_dir.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    fps_save_interval = 1  # Save every 1th frame

    idx = 0
    for video_path in tqdm(list(videos_dir.rglob("*.mp4"))):
        # Open the video file
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            continue

        frame_count = 0

        while True:
            ret, frame = cap.read()

            # Break if no more frames
            if not ret:
                break

            # Save frame at the specified interval
            if frame_count % fps_save_interval == 0:
                # Create filename: image_idx_framecount.jpg
                image_filename = f"image_{idx:06d}_frame_{frame_count:06d}.jpg"
                image_path = images_dir / image_filename

                # Save the frame
                cv2.imwrite(str(image_path), frame)

            frame_count += 1

        # Release the video capture object
        cap.release()
        idx += 1

    print(f"Processed {idx} videos, extracted frames saved to {images_dir}")
