import numpy as np


class Cutout:

    def __init__(self,
                 prob: float = 0.5,
                 area_frac: float = 0.02,
                 aspect_ratio: float = 1 / 0.3,
                 fill_value: float = None,
                 seed: int = 1):
        """
        Cutout augmentation replaces a random rectangular region of the
        spectrogram with mean value / fill value

        :param prob: probability of augmentation, default 0.5
        :param area_frac: fraction of the spectrogram to cut
        :param aspect_ratio: aspect ratio (time / frequencies)
        :param fill_value: fill value. Default None, which means the cut
                           is filled with average value
        :param seed: random seed
        """
        self.prob = prob
        self.area_frac = area_frac
        self.aspect_ratio = aspect_ratio
        self.fill_value = fill_value
        np.random.seed(seed)

    def __call__(self, batch_features: np.ndarray,
                 feature_lengths: np.ndarray) -> np.ndarray:
        return np.stack(
            [self.cut_features(features, feature_length)
             for features, feature_length in
             zip(batch_features, feature_lengths)], axis=0)

    def cut_features(self, features: np.ndarray, feature_length: int):
        pval = np.random.rand()
        if pval > self.prob:
            return features

        features = features.copy()
        _, channels = features.shape
        time = feature_length

        area = time * channels * self.area_frac
        time_cut = int(np.sqrt(area * self.aspect_ratio))
        freq_cut = int(np.sqrt(area / self.aspect_ratio))

        freq_min = np.random.random_integers(
            low=0, high=channels-freq_cut)
        time_min = np.random.random_integers(low=0, high=time-time_cut)

        if self.fill_value:
            fill_value = self.fill_value
        else:
            fill_value = np.mean(features[
                time_min:time_min+time_cut, freq_min:freq_min+freq_cut]
            )

        features[time_min:time_min+time_cut,
                 freq_min:freq_min+freq_cut] = fill_value
        return features
