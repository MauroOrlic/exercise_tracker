import numpy as np
from typing import List
from nptyping import NDArray, Float


class RepCounter:
    COLOR_COUNT = 3
    COLOR_CHANGE_THRESHOLD = 5
    IMAGES_CACHED = 16

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.count_vector_total = self.width * self.height

        self._cache_images: NDArray[NDArray] = np.empty(shape=(0, self.height,self.width, self.COLOR_COUNT))
        self._cache_color_sum: NDArray[NDArray] = np.empty(shape=(0, self.COLOR_COUNT))
        self._cache_moving_pixel_percentage: NDArray[Float] = np.empty(shape=(0,1))

        self.rep_count = 0
        self._last_increment_frames_ago = self.IMAGES_CACHED // 2

        self._np_zeros = np.zeros(3)

    def reset_rep_count(self):
        self.rep_count = 0

    def get_rep_count(self, image: np.ndarray):
        self._cache_images = np.insert(self._cache_images, 0, image, axis=0)
        vector_array = image.reshape(self.count_vector_total, self.COLOR_COUNT)

        self._cache_color_sum = np.insert(self._cache_color_sum, 0, np.sum(vector_array[vector_array != [0.0,0.0,0.0]], axis=0), axis=0)
        self._cache_moving_pixel_percentage = np.insert(self._cache_moving_pixel_percentage, 0, np.count_nonzero(vector_array) / self.count_vector_total, axis=0)

        if len(self._cache_images) == self.IMAGES_CACHED + 1:
            self._cache_images = self._cache_images[:-1]
            self._cache_color_sum = self._cache_color_sum[:-1]
            self._cache_moving_pixel_percentage = self._cache_moving_pixel_percentage[:-1]

            self._calculate_rep_count()

        return np.floor(self.rep_count)

    def _calculate_rep_count(self):
        avg_moving_pixel_perc = np.average(self._cache_moving_pixel_percentage)
        print(avg_moving_pixel_perc*100)

        if  avg_moving_pixel_perc > 0.25:
            half_older = sum(self._cache_color_sum[self.IMAGES_CACHED // 2:]) / (self.IMAGES_CACHED // 2)
            half_newer = sum(self._cache_color_sum[:self.IMAGES_CACHED // 2]) / (self.IMAGES_CACHED // 2)
            allclose = np.allclose(half_older, half_newer, rtol=0.5)
            if not allclose:
                print(allclose)

            if not allclose and self._last_increment_frames_ago >= self.IMAGES_CACHED // 2:
                self.rep_count += np.float(0.5)
                self._last_increment_frames_ago = 0
            else:
                self._last_increment_frames_ago += 1
