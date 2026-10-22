import atexit
import logging
import os
import subprocess

import numpy as np

import cv2

logger = logging.getLogger(__name__)

class VideoWriter:
    def __init__(self, video_path: str, fps: int):
        """Initialize video writer.

        Args:
            video_path: Path to output video file
            fps: Frames per second
        """
        self.video_path = video_path
        self.fps = fps
        self.video_writer = None
        self.needs_transcode = False
        atexit.register(self.release)

    def write(self, frame: np.ndarray):
        if frame is None:
            print(f"No frame to write to video writer; nothing written to file '{self.video_path}'")
            return

        if self.video_writer is None:
            h, w = frame.shape[:2]

            prev_log_level = cv2.getLogLevel()
            cv2.setLogLevel(0)  # suppress C++ errors during codec probe
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (w, h))
            cv2.setLogLevel(prev_log_level)

            if not self.video_writer.isOpened():
                logger.debug("H.264 (avc1) encoder not available, falling back to mp4v")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (w, h))
                self.needs_transcode = True

        # Write frame (convert RGB to BGR for OpenCV)
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            if self.needs_transcode:
                self._transcode_to_h264()

    def _transcode_to_h264(self):
        """Re-encode an mp4v file to H.264 (playable in browsers/VS Code) using imageio-ffmpeg's bundled ffmpeg."""
        try:
            import imageio_ffmpeg
            tmp_path = f"{os.path.splitext(self.video_path)[0]}.h264_tmp.mp4"
            subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", self.video_path,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast",
                 "-movflags", "+faststart", "-threads", "1", tmp_path],
                check=True,
            )
            os.replace(tmp_path, self.video_path)
        except Exception as e:
            logger.warning(f"H.264 transcode failed, keeping mp4v video '{self.video_path}': {e}")

    def __del__(self):
        self.release()
