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


input_video_paths = [
    #'videos/dario_squat_phone.mp4',
    #'videos/dario_squat_webcam.mp4',
    0
]

for input_video_path in input_video_paths:
    if isinstance(input_video_path, str):
        output_video_path = input_video_path[:-4] + '_optical' + input_video_path[-4:]
    else:
        output_video_path = 'webcam_output.mp4'

    capture = VideoCapture(input_video_path)
    # Tinker with parameters depending on video quality and FPS
    rep_counter = RepCounterPoseFlow(
        min_movement_amplitude_per_frame=0.65,
        frames_to_cache=13,
        cached_frames_to_sample=6,  # 1 to frames_to_cache//2
        dot_product_detection_threshold=-0.25  # equal or close to 0.0
    )
    image_processor = DisplayImageProcessor.from_video_capture(capture)
    output = CustomVideoWriter.from_video_capture(capture, file=output_video_path)

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
