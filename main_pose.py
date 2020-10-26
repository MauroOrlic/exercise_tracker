from cv2 import (
    VideoCapture,
    waitKey,
    destroyAllWindows,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT
)

from flow_pose import PoseFlow
from output_utils import DisplayImageProcessor, CustomVideoWriter
from rep_counter import RepCounterPoseFlow


# Captures video from webcam with device index 0
capture = VideoCapture('mauro_squat.mp4')
#capture = VideoCapture(0)
# Tinker with parameters depending on video quality and FPS
rep_counter = RepCounterPoseFlow(
    frames_to_cache=13,  # 2 or more
    cached_frames_to_sample=5,  # 1 to frames_to_cache//2
    dot_product_detection_threshold=0.0  # equal or close to 0.0
)
image_processor = DisplayImageProcessor.from_video_capture(capture)
output = CustomVideoWriter.from_video_capture(capture, file='output.mp4')

print("Press 'Esc' to exit, press 'r' to reset rep count to zero.")
for frame, landmarks, landmarks_flow in PoseFlow.generate_flow_from_capture(capture):

    rep_counter.update_rep_count(landmarks_flow)
    image_processor.display_frame(rep_count=rep_counter.rep_count, frame=frame, landmarks=landmarks)
    output.write(image_processor.current_frame)

    keypress = waitKey(1)
    # Press Esc to exit program
    if keypress == 27:
        break
    # Press 'r' to reset rep count
    elif keypress == ord('r'):
        rep_counter.reset_rep_count()

capture.release()
destroyAllWindows()
output.release()
