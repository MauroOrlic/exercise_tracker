import numpy as np


class RepCounter:
    COLOR_COUNT = 3
    COLOR_CHANGE_THRESHOLD = 5
    IMAGES_CACHED = 8

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.vector_count = self.width * self.height

        self._cache_images: list = list()
        self._cache_color_avg: list = list()

        self._last_diff_was_zeros = True
        self.rep_count = 0

        self._np_zeros = np.zeros(3)

    def reset_rep_count(self):
        self.rep_count = 0

    def get_rep_count(self, image: np.ndarray):
        self._cache_images.insert(0, image)
        self._cache_color_avg.insert(0, np.average(image.reshape(self.vector_count, self.COLOR_COUNT), axis=0))

        if len(self._cache_images) == self.IMAGES_CACHED + 1:
            self._cache_images.pop()
            self._cache_color_avg.pop()

            self._calculate_rep_count()

        return np.floor(self.rep_count)

    def _calculate_rep_count(self):
        diff = (sum(self._cache_color_avg[self.IMAGES_CACHED//2:]) / (self.IMAGES_CACHED //2)) \
               - (sum(self._cache_color_avg[:self.IMAGES_CACHED//2]) / (self.IMAGES_CACHED //2))
        diff[
            np.logical_and(
                -self.COLOR_CHANGE_THRESHOLD < diff,
                diff < self.COLOR_CHANGE_THRESHOLD
            )
        ] = 0

        if not self._last_diff_was_zeros and np.array_equal(diff, self._np_zeros) \
                or self._last_diff_was_zeros and not np.array_equal(diff, self._np_zeros):
            self.rep_count += np.float(0.25)
            self._last_diff_was_zeros = not self._last_diff_was_zeros
