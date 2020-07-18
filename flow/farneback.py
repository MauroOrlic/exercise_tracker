import cv2
from typing import Generator
import numpy as np


def process_frame(frame):
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return processed_frame


def init_frames(capture: cv2.VideoCapture):
    status, frame = capture.read()

    if not status:
        raise IOError('Failed to read frame from capture.')

    hsv_mask = np.zeros_like(frame)
    hsv_mask[..., 1] = 255

    frame_processed = process_frame(frame)

    return frame, frame_processed, hsv_mask


def generate_flow(capture: cv2.VideoCapture) -> Generator[np.ndarray, None, None]:
    frame_current, frame_curr_processed, HSV_MASK = init_frames(capture)

    while True:
        frame_previous = frame_current
        frame_prev_processed = frame_curr_processed

        status, frame_current = capture.read()
        if not status:
            break

        frame_curr_processed = process_frame(frame_current)

        flow = cv2.calcOpticalFlowFarneback(
            prev=frame_prev_processed,
            next=frame_curr_processed,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        HSV_MASK[..., 0] = angle * 180 / np.pi / 2
        magnitude[magnitude < 2] = 0.0
        HSV_MASK[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        frame_output = cv2.cvtColor(HSV_MASK, cv2.COLOR_HSV2BGR)

        yield frame_output
