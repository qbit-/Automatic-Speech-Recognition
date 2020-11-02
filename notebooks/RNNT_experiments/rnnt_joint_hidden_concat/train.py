# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import sys
import os
if os.path.abspath('../') not in sys.path:
    sys.path.append(os.path.abspath('../'))
if os.path.abspath('../../tt_keras') not in sys.path:
    sys.path.append(os.path.abspath('../../tt_keras'))

if os.path.abspath('../../t3f') not in sys.path:
    sys.path.append(os.path.abspath('../../t3f'))

import automatic_speech_recognition as asr
from tensorflow.keras.callbacks import LearningRateScheduler
import time
from tensorflow import keras
import horovod.tensorflow.keras as hvd
from datetime import datetime

# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#       tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#       pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm.notebook import tqdm

# Train/Eval the model

def get_pipeline(model, optimizer=None):
    alphabet = asr.text.Alphabet(lang='en')
    features_extractor = asr.features.TfMFCC(
        features_num=26,
        winlen=0.032,
        winstep=0.02,
    )
    
    if optimizer is None:
        optimizer = tf.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)

    decoder = asr.decoder.GreedyDecoder()
    pipeline = asr.pipeline.RNNTPipeline(
        alphabet, features_extractor, model, optimizer, decoder
    )
    callbacks = []
    return pipeline

dev_dataset = asr.dataset.Audio.from_csv('../../data/dev-clean-index.csv', batch_size=1, use_filesizes=True, librosa_read=False)

alphabet = asr.text.Alphabet(lang='en')
model = asr.model.get_rnnt(26, 
                           num_layers_encoder=8, units_encoder=2048, projection_encoder=640, encoder_reduction_indexes=[1],
                           units_prediction=2048, projection_prediction=640, num_layers_prediction=2, 
                           joint_additional_size=640, joint_aggregation_type='concat',
                           vocab_size=alphabet.size, 
                           blank_label=alphabet.blank_token)
model.load_weights('./rnnt_joint_nohidden_sum_train/rnnt_best.ckpt')

pipeline = get_pipeline(model)
pipeline.compile_model()

folder = './rnnt_joint_nohidden_sum_train'
callbacks = []
# schedule = tf.keras.experimental.CosineDecayRestarts(
#     1e-3, 10, t_mul=2.0, m_mul=1.0, alpha=0.0)
# callbacks.append(LearningRateScheduler(schedule))

callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(folder, 'rnnt_best.ckpt'),
                    monitor='loss', save_weights_only=True,
                    save_best_only=True))

pipeline.fit(dev_dataset, epochs=100, callbacks=callbacks)
pipeline.model.save_weights('./rnnt_weights.h5')