from mediapipe.examples.python.upper_body_pose_tracker import UpperBodyPoseTracker
from cv2 import VideoCapture
from dataclasses import dataclass
from statistics import mean
from typing import Tuple, Generator, Any, Union, Collection, Optional
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

    @classmethod
    def avg_flow(cls, landmark_flows: Collection['LandmarkFlow']):
        return LandmarkFlow(
            mean(lf.x for lf in landmark_flows),
            mean(lf.y for lf in landmark_flows),
            mean(lf.z for lf in landmark_flows),
            mean(lf.visibility for lf in landmark_flows),
        )


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

    @classmethod
    def generate_flow_from_capture(
            cls,
            capture: VideoCapture,
    ) -> Generator[
            Tuple[NDArray, Optional[Landmark], Optional[Tuple[LandmarkFlow, ...]]
        ], Any, Any]:
        tracker = UpperBodyPoseTracker()

        status, frame = capture.read()
        landmarks_current, _ = tracker.run(frame)
        if landmarks_current is not None:
            landmarks_current = Landmark.from_mediapipe_landmarks(landmarks_current)

        if not status:
            raise IOError('Failed to read first frame from capture.')

        while capture.isOpened():
            landmarks_previous = landmarks_current
            status, frame = capture.read()
            if not status:
                break
            landmarks_current, _ = tracker.run(frame)

            flow = None
            if landmarks_current is not None and landmarks_previous is not None:
                landmarks_current = Landmark.from_mediapipe_landmarks(landmarks_current)
                flow = cls._calculate_flow(landmarks_previous, landmarks_current)
            else:
                landmarks_current = None
            yield frame, landmarks_current, flow

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




