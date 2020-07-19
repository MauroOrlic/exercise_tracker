import numpy as np

from nptyping import NDArray, Float


class RepCounter:
    COLOR_COUNT = 3
    COLOR_CHANGE_THRESHOLD = 5
    IMAGES_CACHED = 8
    PERC_MIN_MOVING_PIXELS = 0.25
    DETECTION_RELATIVE_DIFF_THRESHOLD = 0.5

    def __init__(self, width: int, height: int):
        # Sets width and height of video input
        self.width = width
        self.height = height

        # Total number of vectors (or pixels)
        self.count_vector_total = self.width * self.height

        # Initializes caches to empty arrays
        self._cache_images: NDArray[NDArray] = np.empty(shape=(0, self.height,self.width, self.COLOR_COUNT))
        self._cache_color_sum: NDArray[NDArray] = np.empty(shape=(0, self.COLOR_COUNT))
        self._cache_moving_pixel_percentage: NDArray[Float] = np.empty(shape=(0,1))

        self.rep_count = 0

        # Prevents detecting the same change in rapid succession
        self._last_increment_frames_ago = self.IMAGES_CACHED // 2

        self._np_zeros = np.zeros(3)

    def reset_rep_count(self):
        self.rep_count = 0

    def get_rep_count(self, image: np.ndarray):
        # Inserts new frame to the beginnings of the array caches
        self._cache_images = np.insert(self._cache_images, 0, image, axis=0)
        vector_array = image.reshape(self.count_vector_total, self.COLOR_COUNT)
        self._cache_color_sum = np.insert(self._cache_color_sum, 0, np.sum(vector_array[vector_array != self._np_zeros], axis=0), axis=0)
        self._cache_moving_pixel_percentage = np.insert(self._cache_moving_pixel_percentage, 0, np.count_nonzero(vector_array) / self.count_vector_total, axis=0)

        # This part of code doesn't execute unless there are at least 10
        if len(self._cache_images) == self.IMAGES_CACHED + 1:
            # Pops oldest image from array caches
            self._cache_images = self._cache_images[:-1]
            self._cache_color_sum = self._cache_color_sum[:-1]
            self._cache_moving_pixel_percentage = self._cache_moving_pixel_percentage[:-1]

            # Updates rep count if a repetition is detected
            self._calculate_rep_count()

        # Returns current repetition count
        return np.floor(self.rep_count)

    def _calculate_rep_count(self):
        avg_moving_pixel_perc = np.average(self._cache_moving_pixel_percentage)

        # Ignores input if there is little to no pixels moving on screen
        if avg_moving_pixel_perc > self.PERC_MIN_MOVING_PIXELS:
            # Calculates average color (vector direction) for first and second half of cached images
            avg_color_half_older = sum(self._cache_color_sum[self.IMAGES_CACHED // 2:]) / (self.IMAGES_CACHED // 2)
            avg_color_half_newer = sum(self._cache_color_sum[:self.IMAGES_CACHED // 2]) / (self.IMAGES_CACHED // 2)

            color_diff_is_significant = np.allclose(avg_color_half_older, avg_color_half_newer, rtol=self.DETECTION_RELATIVE_DIFF_THRESHOLD)
            # If there is a significant change in color between the older and
            # newer half of cached frames, assume that half a rep was completed.
            # self._last_increment_frames_ago ensures that same action is not detected for multiple iterations in a row
            if not color_diff_is_significant and self._last_increment_frames_ago >= self.IMAGES_CACHED // 2:
                self.rep_count += np.float(0.5)
                self._last_increment_frames_ago = 0
            else:
                self._last_increment_frames_ago += 1
