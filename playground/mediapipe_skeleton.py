import mediapipe as mp
import cv2
from output_utils import CustomVideoWriter
from sys import maxsize
min_x = maxsize
max_x = 0
min_y = maxsize
max_y = 0
min_z = maxsize
max_z = 0
min_visibility = maxsize
max_visibility = 0

cap = cv2.VideoCapture('../mauro_squat.mp4')
#output = CustomVideoWriter.from_video_capture(cap)
print(
    'Press Esc within the output image window to stop the run, or let it '
    'self terminate after 30 seconds.')

tracker = mp.examples.UpperBodyPoseTracker()
while cap.isOpened():
    success, input_frame = cap.read()
    if not success:
        break
    input_frame.flags.writeable = False
    landmarks, annotated_image = tracker.run(input_frame)



    cv2.imshow('MediaPipe upper body pose tracker', annotated_image)
    print(type(landmarks))
    for i in range(len(landmarks.landmark)):
        for flow in landmarks.landmark:
            if flow.x < min_x:
                min_x = flow.x
            if flow.x > max_x:
                max_x = flow.x

            if flow.y < min_y:
                min_y = flow.y
            if flow.y > max_y:
                max_y = flow.y

            if flow.z < min_z:
                min_z = flow.z
            if flow.z > max_z:
                max_z = flow.z

            if flow.visibility < min_visibility:
                min_visibility = flow.visibility
            if flow.visibility > max_visibility:
                max_visibility = flow.visibility
        print(f"X: ({min_x}, {max_x})")
        print(f"Y: ({min_y}, {max_y})")
        print(f"Z: ({min_z}, {max_z})")
        print(f"Visibility: ({min_visibility}, {max_visibility})")

    #output.write(annotated_image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
#output.release()
cv2.destroyAllWindows()
