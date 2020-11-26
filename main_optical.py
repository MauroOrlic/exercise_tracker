import os, errno
import csv
from cv2 import (
    VideoCapture,
    waitKey,
    destroyAllWindows,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT
)

from flow_farneback import generate_flow_from_capture
from output_utils import DisplayImageProcessor, CustomVideoWriter
from rep_counter import RepCounterOpticalFlow


for filename in os.listdir('videos/input'):
    input_video_path = '/'.join(['videos/input', filename])
    output_video_path = '/'.join(['videos/output', filename.split('.')[0] + '_optical.' + filename.split('.')[1]])
    print('Processing to output: ', output_video_path)
    if os.path.isfile(output_video_path):
        print('Output file already exists, skipping!')
        continue

    capture = VideoCapture(input_video_path)
    # Tinker with parameters depending on video quality and FPS
    rep_counter = RepCounterOpticalFlow(
        int(capture.get(CAP_PROP_FRAME_WIDTH)),
        int(capture.get(CAP_PROP_FRAME_HEIGHT)),
        min_nonzero_pixel_percentage=0.05,  # 0.0 to 100.0
        frames_to_cache=13,  # 2 or more
        cached_frames_to_sample=5,  # 1 to frames_to_cache//2
        dot_product_detection_threshold=0.0  # equal or close to 0.0
    )
    image_processor = DisplayImageProcessor.from_video_capture(capture)
    output = CustomVideoWriter.from_video_capture(capture, file=output_video_path)

    print("Press 'Esc' to exit, press 'r' to reset rep count to zero.")
    for magnitude, angle in generate_flow_from_capture(capture):

        rep_counter.update_rep_count(magnitude, angle)
        image_processor.display_frame(
            rep_count=rep_counter.rep_count,
            magnitude=magnitude,
            angle=angle
        )
        output.write(image_processor.current_frame)

        keypress = waitKey(1)
        # Press Esc to exit program
        if keypress == 27:
            break
        # Press 'r' to reset rep count
        elif keypress == ord('r'):
            rep_counter.reset_rep_count()

    with open('results_optical.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        items = list(filename.split('.')[0].split('_'))
        items.append(str(int(rep_counter.rep_count)))
        print(items)
        writer.writerow(items)

    capture.release()
    destroyAllWindows()
    output.release()

print('Finished!')