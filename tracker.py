from flow.farneback import generate_flow
from flow.rep_counter import RepCounter
import cv2


def get_output(capture: cv2.VideoCapture) -> cv2.VideoWriter:
    fps = capture.get(5) // 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    resolution = (int(capture.get(3)), int(capture.get(4)))
    return cv2.VideoWriter('output_of_farneback_webcam.mp4', fourcc, fps, resolution)


# Captures video from webcam
capture = cv2.VideoCapture(0)
# Saves video to current directory
output = get_output(capture)

cv2.namedWindow('Tracker', cv2.WINDOW_AUTOSIZE)

# capture.get(x) for x=3 and x=4 return video width and height, respectively
rep_counter = RepCounter(int(capture.get(3)), int(capture.get(4)))

for image in generate_flow(capture):
    rep_count = rep_counter.get_rep_count(image)
    # Adds text with rep count to image
    cv2.putText(image, f"Rep count: {int(rep_count)}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

    cv2.imshow('Tracker', image)
    output.write(image)

    keypress = cv2.waitKey(1)
    # Press Esc to exit program
    if keypress == 27:
        break
    # Press 'r' to reset rep count
    elif keypress == ord('r'):
        rep_counter.reset_rep_count()

capture.release()
cv2.destroyAllWindows()
output.release()

