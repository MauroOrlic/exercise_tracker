import numpy as np


class RepCounter:

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
        self._cache_avg_vectors = np.insert(self._cache_avg_vectors, 0,
                                            self.get_avg_vector(magnitude, angle),
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
