from flow_pose import PoseFlow
from cv2 import VideoCapture, imshow, waitKey
from mediapipe.examples.python.upper_body_pose_tracker import UpperBodyPoseTracker

#cap = VideoCapture('../mauro_squat_phone.mp4')
cap = VideoCapture(0)

flow_reader = PoseFlow()
tracker = UpperBodyPoseTracker()

while cap.isOpened():
    status, frame = cap.read()

    landmarks = flow_reader.calculate_flow(frame)
    imshow('MyTestWindowe', frame)
    waitKey(1)
    if landmarks is not None:
        print(landmarks[0])




