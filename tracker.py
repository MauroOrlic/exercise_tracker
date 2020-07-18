from flow.farneback import generate_flow
from flow.rep_counter import RepCounter
import cv2

def get_capture(path=None) -> cv2.VideoCapture:
    if path is None:
        capture = cv2.VideoCapture(0)
    else:
        capture = cv2.VideoCapture(path)
    assert capture.isOpened()
    return capture


def get_output(capture: cv2.VideoCapture) -> cv2.VideoWriter:
    fps = capture.get(5)
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    resolution = (int(capture.get(3)), int(capture.get(4)))
    return cv2.VideoWriter('output_of_farneback_webcam.mp4', fourcc, fps, resolution)


rep_counter = RepCounter(640, 480)
capture = get_capture(0)

for image in generate_flow(capture):
    rep_count = rep_counter.get_rep_count(image)
    cv2.putText(image, str(rep_count), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    cv2.imshow('Stvar', image)
    # output.write(frame)
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
# output.release()