from typing import Tuple
import numpy as np


class SpecAugment:

    def __init__(self,
                 F: int = None,
                 mf: int = None,
                 T: int = None,
                 max_p: float = 1.0,
                 mt: int = None,
                 fill_value: float = None,
                 seed: int = 1):
        """ SpecAugment: A Simple Data Augmentation Method.
        This augmentation replaces rectangular regions of the
        spectrogram (across all timesteps or all frequencies)
        by the mean value across all timesteps.

        TODO: implement sparse_image_warp to reproduce the
              original SpecAugment paper

        :param F: maximal range of frequencies to be altered
        :param mf: number of frequency alterations
        :param T: maximal timestep to be altered at random
        :param max_p: upper bound on the timestep: fraction of total timesteps
        :param mt: number of timestep alterations
        :param fill_value: fill value for the masked region.
                           If None, than mean frequency over timesteps is used
        :param seed: random seed
        """
        self.F = F
        self.mf = mf
        self.T = T
        self.mt = mt
        self.max_p = max_p
        self.fill_value = fill_value
        np.random.seed(seed)

    def __call__(self, batch_features: np.ndarray,
                 feature_lengths: np.ndarray) -> np.ndarray:
        return np.stack(
            [self.mask_features(features, feature_length)
             for features, feature_length in
             zip(batch_features, feature_lengths)], axis=0)

    def mask_features(self, features: np.ndarray,
                      feature_length: int) -> np.ndarray:
        features = features.copy()
        _, channels = features.shape
        time = feature_length

        if not self.fill_value:
            # The mean should be zero if features are normalized
            fill = features.mean(axis=0)
        else:
            fill = np.zeros(features.shape[1])

        if self.F and self.mf:
            features = self.mask_frequencies(
                features, fill, channels, self.F, self.mf)
        if self.T and self.mt:
            features = self.mask_time(
                features, fill, time, self.T, self.mt, self.max_p)
        return features

    @staticmethod
    def mask_frequencies(features: np.ndarray, fill: np.ndarray,
                         channels: int, F: int, mf: int):
        for i in range(mf):
            f = np.random.random_integers(low=0, high=F)
            f0 = np.random.random_integers(low=0, high=channels-F)
            features[:, f0:f0+f] = fill[f0:f0+f]
        return features

    @staticmethod
    def mask_time(features: np.ndarray, fill: np.ndarray,
                  time: int, T: int, mt: int, max_p: float = 1.0):
        for i in range(mt):
            t = int(np.clip(np.random.random_integers(low=0, high=T),
                            a_min=0, a_max=max_p * time))
            t0 = np.random.random_integers(low=0, high=time-T)
            features[t0:t0+t, :] = fill
        return features
