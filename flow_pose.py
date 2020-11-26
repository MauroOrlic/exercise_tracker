from mediapipe.examples.python.upper_body_pose_tracker import UpperBodyPoseTracker
from cv2 import VideoCapture
from dataclasses import dataclass
from statistics import mean
from typing import Tuple, Generator, Any, Union, Optional,  List
from nptyping import NDArray
import numpy as np


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float

    @classmethod
    def from_mediapipe_landmarks(cls, landmarks) -> Tuple['Landmark']:
        return tuple(
            Landmark(
                data_point.x, data_point.y, data_point.z,
                data_point.visibility
            )
            for data_point in landmarks.landmark
        )

    def as_vector(self):
        return np.ndarray([self.x, self.y, self.z])


@dataclass
class LandmarkFlow:
    x: float
    y: float
    z: float
    visibility: float

    @classmethod
    def from_landmarks(cls, position_previous: Landmark, postition_current: Landmark):
        return LandmarkFlow(
            (postition_current.x - position_previous.x) * 100,
            (postition_current.y - position_previous.y) * 100,
            (postition_current.z - position_previous.z) * 100,
            mean((postition_current.visibility, position_previous.visibility))
        )

    def as_np_vector(self):
        return np.array([self.x, self.y])


class PoseFlow:
    def __init__(
            self
    ):
        self._landmarks_previous = None
        self._landmarks_current = None
        self._tracker = UpperBodyPoseTracker()

    @classmethod
    def from_capture(cls, capture: VideoCapture):
        pass

    @staticmethod
    def mean_landmark_group(landmarks_group: List[Tuple[Landmark]]) -> Optional[Tuple[Landmark]]:
        if len(landmarks_group) == 0:
            return None
        landmark_count = len(landmarks_group[0])

        return tuple(
            Landmark(
                mean(landmarks[i].x for landmarks in landmarks_group),
                mean(landmarks[i].y for landmarks in landmarks_group),
                mean(landmarks[i].z for landmarks in landmarks_group),
                mean(landmarks[i].visibility for landmarks in landmarks_group)
            )
            for i in range(landmark_count)
        )


    @classmethod
    def generate_flow_from_capture(
            cls,
            capture: VideoCapture,
            smoothing: int = 3
    ) -> Generator[
        Tuple[NDArray, Optional[Landmark], Optional[Tuple[LandmarkFlow, ...]]]
        , Any, Any]:
        tracker = UpperBodyPoseTracker()

        landmarks_previous = None
        landmarks_current = None

        landmarks_group_previous = []
        landmarks_group_current = []

        while capture.isOpened():
            if landmarks_current is not None:
                landmarks_previous = landmarks_current
                landmarks_group_previous.insert(0, landmarks_previous)
                if len(landmarks_group_previous) > smoothing:
                    landmarks_group_previous.pop()

            status, frame = capture.read()
            if not status:
                print(f"Capture status returned: {status}")
                break
            landmarks_current, _ = tracker.run(frame)
            if landmarks_current is not None:
                landmarks_current = Landmark.from_mediapipe_landmarks(landmarks_current)
                landmarks_group_current.insert(0, landmarks_current)
                if len(landmarks_group_current) > smoothing:
                    landmarks_group_current.pop()

            if landmarks_current is None or landmarks_previous is None:
                yield frame, None, None
            else:
                landmarks_current_mean = cls.mean_landmark_group(landmarks_group_current)
                landmarks_previous_mean = cls.mean_landmark_group(landmarks_group_previous)
                flow = cls._calculate_flow(
                    landmarks_previous_mean,
                    landmarks_current_mean
                )
                yield frame, landmarks_current_mean, flow

    @classmethod
    def _calculate_flow(
            cls,
            landmarks_previous: Tuple[Landmark],
            landmarks_current: Tuple[Landmark]
    ):
        count_landmarks_previous = len(landmarks_previous)
        count_landmarks_current = len(landmarks_current)
        if not count_landmarks_previous == count_landmarks_current:
            raise ValueError(
                f"Previous and current landmark sets are of different lengths: "
                f"{count_landmarks_previous} and {count_landmarks_current}"
            )

        return tuple(
            LandmarkFlow.from_landmarks(landmarks_previous[i], landmarks_current[i])
            for i in range(len(landmarks_previous))
        )

    def calculate_flow(self, next_frame: NDArray) -> Union[Tuple[LandmarkFlow], None]:
        self._landmarks_previous = self._landmarks_current

        landmarks, _ = self._tracker.run(next_frame)
        if landmarks is None:
            return None
        self._landmarks_current = Landmark.from_mediapipe_landmarks(landmarks)

        if self._landmarks_previous is None:
            return self._calculate_flow(self._landmarks_current, self._landmarks_current)
        else:
            return self._calculate_flow(self._landmarks_previous, self._landmarks_current)




