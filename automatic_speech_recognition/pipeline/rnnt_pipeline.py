import os
from types import MethodType
import logging
from typing import List, Callable, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
import collections.abc
from . import Pipeline
from .. import augmentation
from . import CTCPipeline
from ..features import FeaturesExtractor

logger = logging.getLogger('asr.pipeline')

try:
    from warprnnt_tensorflow import rnnt_loss
except:
    logger.info("Could not import warp-rnnt loss")
    

class RNNTPipeline(CTCPipeline):
    """
    Pipeline modifies preprocessing step to feed labels as model inputs as well.
    """

    def preprocess(self,
                   batch: Tuple[List[np.ndarray], List[str]],
                   is_extracted: bool,
                   augmentation: augmentation.Augmentation):
        """ Preprocess batch data to format understandable to a model. """
        data, transcripts = batch
        if is_extracted:  # then just align features
            feature_lengths = [len(feature_seq) for feature_seq in data]
            features = FeaturesExtractor.align(data)
        else:
            features, feature_lengths = self._features_extractor(data)
        features = augmentation(features) if augmentation else features
        feature_lengths = np.array(feature_lengths)

        labels = self._alphabet.get_batch_labels(transcripts)
        label_lengths = np.array([len(decoded_text) for decoded_text in transcripts])
        
        return (features, labels, feature_lengths, label_lengths),

    def predict(self, batch_audio: List[np.ndarray], **kwargs) -> List[str]:
        """ Get ready features, and make a prediction. All additional model inputs
        except for the first one are replaced by zero tensors
        Implementation from https://github.com/zhanghaobaba/RNN-Transducer/blob/786fa75ff65c8ce859183d3c67aa408ff7fdef13/model.py#L33
        """
        features, feature_lengths = self._features_extractor(batch_audio)
        texts = []
        
        for feature_vector, length in zip(features, feature_lengths):
            features_vector = feature_vector[:length]
            decoded_labels = self.decoder(features_vector)
            texts.append(self._alphabet.get_batch_transcripts([decoded_labels])[0])
        
        return texts

    def get_loss(self) -> Callable:
        """ The CTC loss using rnnt_loss. """
        def a(*args):
            return 0
        return a