import numpy as np
import time


class RepCounter:
    IMAGES_CACHED = 11
    THING = 4
    assert 2*THING <= IMAGES_CACHED
    VECTOR_COMPONENT_COUNT = 2

    vectorize_component_x = np.vectorize(np.cos, otypes=[np.single])
    vectorize_component_y = np.vectorize(np.sin, otypes=[np.single])

    def __init__(self, width: int, height: int):
        # Sets width and height of video input
        self.width = width
        self.height = height

        # Total number of vectors (or pixels)
        self.count_vector_total = self.width * self.height

        # Initializes caches to empty arrays
        self._cache_avg_vectors = np.empty(shape=(0, self.VECTOR_COMPONENT_COUNT))

        self.rep_count = 0

        # Prevents detecting the same change in rapid succession
        self._last_increment_frames_ago = self.IMAGES_CACHED // 2

    def reset_rep_count(self):
        self.rep_count = 0

    def get_avg_vector(self, magnitude: np.ndarray, angle: np.ndarray):
        return np.add.reduce(
            np.reshape(
                np.dstack((
                    np.cos(angle) * magnitude,
                    np.sin(angle) * magnitude
                )),
                (self.width * self.height, 2)
            )
        ) / (self.width * self.height)

    def get_rep_count(self, magnitude: np.ndarray, angle: np.ndarray):
        # Inserts new frame to the beginnings of the array caches

        avg_vector = self.get_avg_vector(magnitude, angle)
        self._cache_avg_vectors = np.insert(
            self._cache_avg_vectors,
            0,
            avg_vector,
            axis=0
        )

        # This part of code doesn't execute until there are self.IMAGES_CACHED items cached
        if len(self._cache_avg_vectors) == self.IMAGES_CACHED + 1:
            self._cache_avg_vectors = self._cache_avg_vectors[:-1]

            # Updates rep count if a repetition is detected
            self._calculate_rep_count()

        # Returns current repetition count
        return np.floor(self.rep_count)

    def _calculate_rep_count(self):
        if self._last_increment_frames_ago >= self.IMAGES_CACHED and \
                np.dot(
                    np.add.reduce(self._cache_avg_vectors[:self.THING]) / self.THING,
                    np.add.reduce(self._cache_avg_vectors[-self.THING:]) / self.THING
                ) < 0:
            self.rep_count += 0.5
            self._last_increment_frames_ago = 0
        else:
            self._last_increment_frames_ago += 1




