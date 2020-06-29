import numpy as np


class Cutout:

    def __init__(self,
                 F: int = None,
                 T: int = None,
                 n: int = None,
                 fill_value: float = 0,
                 seed: int = 1):
        """
        Cutout augmentation replaces a random rectangular region of the
        spectrogram with mean value / fill value

        :param F: maximal range of frequencies to be altered
        :param T: maximal timestep to be altered at random
        :param n: number of cutout regions
        :param fill_value: fill value. Default None, which means the cut
                           is filled with average value
        :param seed: random seed
        """
        self.F = F
        self.T = T
        self.n = n
        self.fill_value = fill_value
        np.random.seed(seed)

    def __call__(self, batch_features: np.ndarray,
                 feature_lengths: np.ndarray) -> np.ndarray:
        return np.stack(
            [self.cut_features(features, feature_length,
                               self.F, self.T, self.n, self.fill_value)
             for features, feature_length in
             zip(batch_features, feature_lengths)], axis=0)

    @staticmethod
    def cut_features(features: np.ndarray, feature_length: int, F: int,
                     T: int, n_cuts: int, fill_value: float = None):
        features = features.copy()
        _, channels = features.shape
        time = feature_length

        for i in range(n_cuts):
            t = np.random.random_integers(low=0, high=T)
            t0 = np.random.random_integers(low=0, high=time-T)

            f = np.random.random_integers(low=0, high=F)
            f0 = np.random.random_integers(low=0, high=channels-F)

            if not fill_value:
                fill_value = np.mean(features[t0:t0+t, f0:f0+f])

            features[t0:t0+t, f0:f0+f] = fill_value
        return features
