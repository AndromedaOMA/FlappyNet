from collections import deque
import PIL.Image as Image
import numpy as np


def preprocess_frame(frame, height=84, width=84):
    gray = Image.fromarray(frame).convert('L')
    resized = gray.resize((height, width))
    normalized = np.array(resized, dtype=np.float32) / 255.0
    frame = np.expand_dims(normalized, axis=0)               # Add channel dimension [1, H, W]
    return frame


class FrameStacker:
    def __init__(self, stack_size=4, height=84, width=84):
        self.stack_size = stack_size
        self.height = height
        self.width = width
        self.frames = deque(maxlen=stack_size)

    def reset(self, frame):
        processed_frame = preprocess_frame(frame, self.height, self.width)
        for _ in range(self.stack_size):
            self.frames.append(processed_frame)
        return np.concatenate(self.frames, axis=0)

    def update(self, frame):
        processed_frame = preprocess_frame(frame, self.height, self.width)
        self.frames.append(processed_frame)
        return np.concatenate(self.frames, axis=0)
