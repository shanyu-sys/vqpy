import cv2
from loguru import logger
from vqpy.backend.operator.base import Operator
from vqpy.backend.frame import Frame
import numpy as np

class VideoReader(Operator):
    def __init__(self, video_path: str):
        self._cap = cv2.VideoCapture(video_path)
        self.frame_id = -1
        self.metadata = self.get_metadata()
    
    def get_metadata(self):
        frame_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        frame_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Metadata of video is width={frame_width}, \
                      height={frame_height}, fps={fps}, n_frames={n_frames}")
        metadata = dict(
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            n_frames=n_frames,
        )
        return metadata

    def has_next(self) -> bool:
        if self.frame_id + 1 < self.metadata["n_frames"]:
            return True
        else:
            self.close()
            return False
    

    def next(self) -> Frame:
        if self.has_next():
            self.frame_id += 1
            ret_val, frame_image = self._cap.read()
            frame = Frame(video_metadata=self.metadata,
                          id=self.frame_id,
                          image=frame_image)
            return frame
        else:
            raise StopIteration

    def close(self):
        self._cap.release()

import time
class VideoReaderBatchLoad(Operator):
    def __init__(self, video_path: str, fps=None, size=None, limit=None):
        self._cap = cv2.VideoCapture(video_path)
        self.frame_id = 0
        self.metadata = self.get_metadata()
        self.size = size
        self.video = None
        self.fps=fps if fps is not None else self.metadata["fps"]
        self.original_fps = self.metadata["fps"]
        print(self.original_fps)
        self.n_skip = int(self.original_fps // self.fps)
        self.limit = limit

    
    def _load_video_to_numpy(self):
        
        result = []
        frame_id = 0
        while True:
            ret_val, frame_image = self._cap.read()
            if not ret_val:
                break
            # print(ret_val)
            # print(frame_image)
            if frame_id % self.n_skip == 0:
                if self.size is not None:
                    frame_image = cv2.resize(frame_image, self.size)
                result.append(frame_image)
            frame_id += 1
            if self.limit is not None and frame_id > self.limit:
                break

        # self.video = np.stack(result, axis=0)
        self.video = result
        

    def get_metadata(self):
        frame_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        frame_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Metadata of video is width={frame_width}, \
                      height={frame_height}, fps={fps}, n_frames={n_frames}")
        metadata = dict(
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            n_frames=n_frames,
        )
        return metadata

    def has_next(self) -> bool:
        if self.video is None:
            t1 = time.perf_counter()
            self._load_video_to_numpy()
            t2 = time.perf_counter()
            print(f"load video using {t2 - t1}")
        if self.frame_id < len(self.video):
            return True
        else:
            self.close()
            return False

    def next(self, n=1) -> Frame:
        # result = []
        if self.has_next():
            frame_image = self.video[self.frame_id]
            frame = Frame(video_metadata=self.metadata,
                          id=self.frame_id,
                          image=frame_image)
            self.frame_id += 1
            return frame
        else:
            raise StopIteration

    def close(self):
        self._cap.release()
