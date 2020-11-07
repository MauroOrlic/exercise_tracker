import numpy as np
from flow_pose import LandmarkFlow, Landmark
from typing import Tuple
from statistics import mean
from collections import deque
from typing import Tuple, List


class RepCounterOpticalFlow:

    VECTOR_COMPONENT_COUNT = 2

    def __init__(
            self,
            width: int,
            height: int,
            min_nonzero_pixel_percentage=0.15,
            frames_to_cache=13,
            cached_frames_to_sample=5,
            dot_product_detection_threshold=0.0
    ):
        """
        Given a stream of optical flow magnitude and angle arrays, counts the number of repetitions of an exercise.

        :param width: width of the video source in pixels
        :param height: width of the video source in pixels
        :param min_nonzero_pixel_percentage: ignores frames that have low percentage of pixels moving
        :param frames_to_cache: number of last x frames to cache, smooths detection. Increase if too
            many reps are being detected, if it takes you x frames to do a single rep, make sure this 
            value is set to less than that.
        :param cached_frames_to_sample: Keep this value at half of frames_to_cache. If changes in direction
            of movement during exercise are sluggish and are not getting detected try decreasing this value.
        :param dot_product_detection_threshold: Keep this around 0.0.
        """
        # Sets width and height of video input
        self._width = width
        self._height = height
        # Total number of vectors/pixels
        self._count_vector_total = self._width * self._height

        # Configuring various properties
        self.min_nonzero_pixel_percentage = min_nonzero_pixel_percentage
        self.frames_to_cache = frames_to_cache
        self.cached_frames_to_sample = cached_frames_to_sample
        self.dot_product_detection_threshold = dot_product_detection_threshold

        # Initializes cache for average vectors (an avg vector is calculated for each frame received)
        self._cache_avg_vectors = np.empty(shape=(0, self.VECTOR_COMPONENT_COUNT))

        self._rep_count = 0

        # Prevents detecting the same change of movement multiple frames in a row
        self._last_increment_frames_ago = 0

    @property
    def min_nonzero_pixel_percentage(self):
        return self._min_nonzero_vector_percentage

    @min_nonzero_pixel_percentage.setter
    def min_nonzero_pixel_percentage(self, v: float):
        if not isinstance(v, (int, float)):
            raise TypeError(f"expected {float} or {int}, not {type(v)}")
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"expected value between 0 and 1, not {v}")
        self._min_nonzero_vector_percentage = v

    @property
    def frames_to_cache(self):
        return self._images_to_cache

    @frames_to_cache.setter
    def frames_to_cache(self, v: int):
        if not isinstance(v, int):
            raise TypeError(f"expected {int}, not {type(v)}")
        if not 2 <= v:
            raise ValueError(f"expected value greater or equal to 2, not {v}")
        self._images_to_cache = v

    @property
    def cached_frames_to_sample(self):
        return self._cached_frames_to_sample

    @cached_frames_to_sample.setter
    def cached_frames_to_sample(self, v: int):
        if not isinstance(v, int):
            raise TypeError(f"expected {int}, not {type(v)}")
        if not 1 <= v <= self.frames_to_cache // 2:
            raise ValueError(f"expected value between 1 and {self.frames_to_cache // 2} "
                             f"(half of 'frames_to_cache' rounded down), not {v}")
        self._cached_frames_to_sample = v

    @property
    def dot_product_detection_threshold(self):
        return self._dot_product_detection_threshold

    @dot_product_detection_threshold.setter
    def dot_product_detection_threshold(self, v: float):
        if not isinstance(v, (int, float)):
            raise TypeError(f"expected {float} or {int}, not {type(v)}")
        self._dot_product_detection_threshold = v

    @property
    def rep_count(self) -> int:
        return np.floor(self._rep_count)

    def reset_rep_count(self):
        self._rep_count = 0

    def get_avg_vector(self, magnitude: np.ndarray, angle: np.ndarray):
        # Converts magnitude and angle 2D arrays to a single 2D array containing [x, y] vectors,
        # then calculates and returns the avg [x, y] vector
        return np.add.reduce(
            np.reshape(
                np.dstack((
                    np.cos(angle) * magnitude,
                    np.sin(angle) * magnitude
                )),
                (self._count_vector_total, self.VECTOR_COMPONENT_COUNT)
            )
        ) / self._count_vector_total

    def update_rep_count(self, magnitude: np.ndarray, angle: np.ndarray):
        # Ignores low/no activity frames (useful when subject is resting for a few frames at the end of a rep)
        if np.count_nonzero(magnitude) / self._count_vector_total <= self.min_nonzero_pixel_percentage:
            return

        # Calculates average vector for the frame and inserts it to the beginning of the self._cache_avg_vectors cache
        avg_vector = self.get_avg_vector(magnitude, angle)
        if np.isnan(avg_vector[0]):
            print(magnitude, angle)
            print(avg_vector)
        self._cache_avg_vectors = np.insert(self._cache_avg_vectors, 0,
                                            avg_vector,
                                            axis=0)

        # This part of code doesn't execute until there are self.frames_to_cache items cached
        if len(self._cache_avg_vectors) == self.frames_to_cache + 1:
            self._cache_avg_vectors = self._cache_avg_vectors[:-1]

            # Updates rep count if a repetition is detected
            self._calculate_rep_count()

    def _calculate_rep_count(self):
        # Prevents detecting the same change of movement multiple frames in a row
        if self._last_increment_frames_ago >= self.frames_to_cache - self.cached_frames_to_sample:
            # Takes the 'oldest' and 'newest' self.cached_frames_to_sample number of vectors,
            # calculates the average vector for the 'oldest' and 'newest' group of vectors (smoothing)
            # and then calculates the dot product of the two vectors.
            dot_product = np.dot(
                np.add.reduce(self._cache_avg_vectors[:self.cached_frames_to_sample]) / self.cached_frames_to_sample,
                np.add.reduce(self._cache_avg_vectors[-self.cached_frames_to_sample:]) / self.cached_frames_to_sample
            )
            # If the dot product is negative that means the subject suddenly changed the
            # direction of movement (we interpret this as doing half of a rep)
            if dot_product < self.dot_product_detection_threshold:
                self._rep_count += 0.5
                self._last_increment_frames_ago = 0
        else:
            self._last_increment_frames_ago += 1


class RepCounterPoseFlow:
    MARKERS_PER_FRAME = 31
    MARKER_VECTOR_COMPONENT_COUNT = 3

    def __init__(
            self,
            min_movement_amplitude_per_frame=0.65,
            frames_to_cache=13,
            cached_frames_to_sample=5,
            dot_product_detection_threshold=-0.25
    ):
        self.frames_to_cache = frames_to_cache
        self.cached_frames_to_sample = cached_frames_to_sample
        self.dot_product_detection_threshold = dot_product_detection_threshold

        self._rep_count = 0

        self._cached_frames_landmarks: List[Tuple[LandmarkFlow, ...]] = list()

        self._last_increment_frames_ago = 0

    @property
    def frames_to_cache(self):
        return self._images_to_cache

    @frames_to_cache.setter
    def frames_to_cache(self, v: int):
        if not isinstance(v, int):
            raise TypeError(f"expected {int}, not {type(v)}")
        if not 2 <= v:
            raise ValueError(f"expected value greater or equal to 2, not {v}")
        self._images_to_cache = v

    @property
    def cached_frames_to_sample(self):
        return self._cached_images_to_sample

    @cached_frames_to_sample.setter
    def cached_frames_to_sample(self, v: int):
        if not isinstance(v, int):
            raise TypeError(f"expected {int}, not {type(v)}")
        if not 1 <= v <= self.frames_to_cache // 2:
            raise ValueError(f"expected value between 1 and {self.frames_to_cache // 2} "
                             f"(half of 'frames_to_cache' rounded down), not {v}")
        self._cached_images_to_sample = v

    @property
    def dot_product_detection_threshold(self):
        return self._dot_product_detection_threshold

    @dot_product_detection_threshold.setter
    def dot_product_detection_threshold(self, v: float):
        if not isinstance(v, (int, float)):
            raise TypeError(f"expected {float} or {int}, not {type(v)}")
        self._dot_product_detection_threshold = v

    @property
    def rep_count(self) -> int:
        return np.floor(self._rep_count)

    def reset_rep_count(self):
        self._rep_count = 0

    def _frame_avg_flow_magnitude(self, frame_flow: Tuple[LandmarkFlow]) -> float:
        avg_landmark = LandmarkFlow(0, 0, 0, 0)
        for landmark_flow in frame_flow:
            avg_landmark.x += landmark_flow.x / self.MARKERS_PER_FRAME
            avg_landmark.y += landmark_flow.y / self.MARKERS_PER_FRAME
            avg_landmark.z += landmark_flow.z / self.MARKERS_PER_FRAME
            avg_landmark.visibility += landmark_flow.visibility / self.MARKERS_PER_FRAME
        avglandmark_vector = avg_landmark.as_np_vector()
        return np.sqrt(avglandmark_vector.dot(avglandmark_vector))

    def update_rep_count(self, landmarks_flow: Tuple[LandmarkFlow]):
        if landmarks_flow is None or self._frame_avg_flow_magnitude(landmarks_flow) < 0.5:
            return

        print(self._frame_avg_flow_magnitude(landmarks_flow))

        self._cached_frames_landmarks.insert(0, landmarks_flow)

        if len(self._cached_frames_landmarks) == self.frames_to_cache + 1:
            self._cached_frames_landmarks.pop()

            self._calculate_rep_count()

    def _calculate_rep_count(self):
        # Prevents detecting the same change of movement multiple frames in a row
        if self._last_increment_frames_ago >= self.frames_to_cache - self.cached_frames_to_sample:

            previous = self._cached_frames_landmarks[:self.cached_frames_to_sample]
            previous_avg_frame = tuple(LandmarkFlow(0, 0, 0, 0) for i in range(self.MARKERS_PER_FRAME))
            previous_frame_count = len(previous)

            for frame_landmarks_flow in previous:
                for i in range(previous_frame_count):
                    previous_avg_frame[i].x += frame_landmarks_flow[i].x / previous_frame_count
                    previous_avg_frame[i].y += frame_landmarks_flow[i].y / previous_frame_count
                    previous_avg_frame[i].z += frame_landmarks_flow[i].z / previous_frame_count
                    previous_avg_frame[i].visibility += frame_landmarks_flow[i].visibility / previous_frame_count

            current = self._cached_frames_landmarks[-self.cached_frames_to_sample:]
            current_avg_frame = [LandmarkFlow(0, 0, 0, 0) for i in range(self.MARKERS_PER_FRAME)]
            current_frame_count = len(current)

            for frame_landmarks_flow in current:
                for i in range(previous_frame_count):
                    current_avg_frame[i].x += frame_landmarks_flow[i].x / current_frame_count
                    current_avg_frame[i].y += frame_landmarks_flow[i].y / current_frame_count
                    current_avg_frame[i].z += frame_landmarks_flow[i].z / current_frame_count
                    current_avg_frame[i].visibility += frame_landmarks_flow[i].visibility / current_frame_count

            frame_marker_dot_products = [
                np.dot(
                    previous_avg_frame[i].as_np_vector(),
                    current_avg_frame[i].as_np_vector()
                )
                for i in range(self.MARKERS_PER_FRAME)]

            if sum(frame_marker_dot_products) < self.dot_product_detection_threshold:
                self._rep_count += 0.5
                self._last_increment_frames_ago = 0
        else:
            self._last_increment_frames_ago += 1
