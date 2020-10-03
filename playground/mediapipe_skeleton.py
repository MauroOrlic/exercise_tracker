import mediapipe as mp

pose_tracker = mp.examples.UpperBodyPoseTracker()
pose_tracker.run_live()