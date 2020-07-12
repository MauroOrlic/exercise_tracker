import cv2
import numpy as np


def get_capture() -> cv2.VideoCapture:
    #capture = cv2.VideoCapture('mauro_squat.mp4')
    capture = cv2.VideoCapture(0)
    assert capture.isOpened()
    return capture


def get_output(capture: cv2.VideoCapture) -> cv2.VideoWriter:
    fps = capture.get(5)
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    resolution = (int(capture.get(3)), int(capture.get(4)))
    return cv2.VideoWriter('output_of_farneback_webcam.mp4', fourcc, fps, resolution)


def write_output(output: cv2.VideoWriter, image):
    return
    output.write(image)


def process_frame(frame):
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Processed', processed_frame)
    return processed_frame


def init_frames(capture: cv2.VideoCapture):
    status, frame = capture.read()

    if not status:
        raise IOError('Failed to read frame from capture.')

    hsv_mask = np.zeros_like(frame)
    hsv_mask[..., 1] = 255

    frame_processed = process_frame(frame)

    return frame, frame_processed, hsv_mask

def get_flow(capture: cv2.VideoCapture):
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
        magnitude[magnitude < 5] = 0.0
        #cv2.imshow('Magnitude', magnitude)
        HSV_MASK[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        final_display = cv2.cvtColor(HSV_MASK, cv2.COLOR_HSV2BGR)
        #cv2.imshow('Final', final_display)
        write_output(output, final_display)

        if cv2.waitKey(1) == 27:
            break
    capture.release()


if __name__ == '__main__':
    capture = get_capture()
    output = get_output(capture)
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
        magnitude[magnitude < 5] = 0.0
        #cv2.imshow('Magnitude', magnitude)
        HSV_MASK[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        final_display = cv2.cvtColor(HSV_MASK, cv2.COLOR_HSV2BGR)
        cv2.imshow('Final', final_display)
        write_output(output, final_display)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    capture.release()
    if output is not None:
        output.release()

