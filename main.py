from flow_farneback import generate_flow_from_capture
from rep_counter import RepCounter
from output_utils import DisplayImageProcessor, CustomVideoWriter
import cv2


def main():
    # Captures video from webcam with device index 0
    capture = cv2.VideoCapture(0)

    rep_counter = RepCounter.from_video_capture(capture)
    image_processor = DisplayImageProcessor.from_video_capture(capture)
    output = CustomVideoWriter.from_video_capture(capture)

    for magnitude, angle in generate_flow_from_capture(capture):

        rep_counter.update_rep_count(magnitude, angle)
        image_processor.display_frame(magnitude, angle, rep_counter.rep_count)
        output.write(image_processor.current_frame)

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


if __name__ == '__main__':
    main()
