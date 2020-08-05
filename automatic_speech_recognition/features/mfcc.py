import math
import numpy as np
import librosa
from . import audio_utils
from .. import features
from tensorflow.python.ops import gen_audio_ops as contrib_audio


class TfMFCC(features.FeaturesExtractor):

    def __init__(self, features_num: int, standardize=None, sample_rate=16000, 
    winlen=0.032, winstep=0.02):
        super().__init__(standardize=standardize)
        self.features_num = features_num
        self.sample_rate = sample_rate
        self.window_size = int(winlen * sample_rate)
        self.window_step = int(winstep * sample_rate)

    def make_features(self, audio: np.ndarray) -> np.ndarray:
        """ Use `python_speech_features` lib to extract log filter banks from
        the features file. """
        spectrogram = contrib_audio.audio_spectrogram(audio.audio,
                                                    window_size=self.window_size,
                                                    stride=self.window_step,
                                                    magnitude_squared=True)
        mfccs = contrib_audio.mfcc(spectrogram=spectrogram,
                                    sample_rate=self.sample_rate,
                                    dct_coefficient_count=self.features_num,
                                    upper_frequency_limit=self.sample_rate//2)
        return mfccs[0]


class MFCC(features.FeaturesExtractor):
    """
    This class calculates the Mel-frequency Cepstral Coefficients (MFCCs)
    The procedure is described in:
    https://haythamfayek.com/2016/04/21/
    speech-processing-for-machine-learning.html
    """
    def __init__(self, features_num: int, sample_rate: int = 16000,
                 winlen: float = 0.02, winstep: float = 0.01,
                 window="hann", n_fft=None,
                 dct_type: int = 2, lifter: float = 0,
                 dither: float = 1e-5, preemph: float = 0.97,
                 standardize="per_feature"):

        self.features_num = features_num
        self.sample_rate = sample_rate
        self.win_length = math.ceil(winlen * sample_rate)
        self.hop_length = math.ceil(winstep * sample_rate)
        self.window = window
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.dct_type = dct_type
        self.lifter = lifter
        self.dither = dither
        self.preemph = preemph
        super().__init__(standardize=standardize)

    def make_features(self, audio: np.ndarray) -> np.ndarray:
        """ Extract MFCCs from the audio. """
        # dither
        if self.dither > 0:
            audio = audio_utils.dither(audio, self.dither)

        # do preemphasis
        if self.preemph is not None:
            audio = audio_utils.preemphasize(audio, self.preemph)

        # get mfccs. Librosa returns (n_mfcc, t)
        features = librosa.feature.mfcc(
            audio, self.sample_rate, n_mfcc=self.features_num,
            dct_type=self.dct_type, lifter=self.lifter, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.win_length,
            window=self.window
        )

        # put features into correct order (time, n_features)
        return features.transpose()
