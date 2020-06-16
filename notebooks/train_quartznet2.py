import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
import tensorflow_addons as tfa
import sys
import os
if os.path.abspath('../') not in sys.path:
    sys.path.append(os.path.abspath('../'))

import automatic_speech_recognition as asr
from automatic_speech_recognition.utils import wrap_call_methods
import time
from datetime import datetime
import argparse
import pickle

import numpy as np

from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import logging
from typing import List

logger = logging.getLogger('asr.pipeline')


def distribute_model(model: keras.Model, gpus: List[str]) -> keras.Model:
    """ Replicates a model on different GPUs. """
    try:
        dist_model = keras.utils.multi_gpu_model(model, len(gpus))
        logger.info("Training using multiple GPUs")
    except ValueError:
        dist_model = model
        logger.info("Training using single GPU or CPU")
    return dist_model

def get_pipeline(model, optimizer=None):
    alphabet = asr.text.Alphabet(lang='en')
    features_extractor = asr.features.FilterBanks(
        features_num=64,
        sample_rate=16000,
        standardize="per_feature",
        winlen=0.02,
        winstep=0.01,
    )
    if not optimizer:
        optimizer = tf.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
    decoder = asr.decoder.GreedyDecoder()
    pipeline = asr.pipeline.CTCPipeline(
        alphabet, features_extractor, model, optimizer, decoder
    )
    return pipeline


def train_model(filename, dataset_idx, n_blocks=1, is_mixed_precision=True,
                val_dataset_idx=None, batch_size=10, epochs=700,
                tensorboard=False,
                restart_filename=None):

    basename = os.path.basename(filename).split('.')[0]
    model_dir = os.path.join(os.path.dirname(filename), basename + '_train')
    os.makedirs(model_dir, exist_ok=True)

    model = asr.model.get_quartznet(64, 29, num_b_block_repeats=n_blocks,
                                    is_mixed_precision=is_mixed_precision)

    if restart_filename:
        model.load_weights(restart_filename)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    dist_model = distribute_model(model, gpus)

    dataset = asr.dataset.Audio.from_csv(dataset_idx, batch_size=batch_size,
                                         use_filesizes=True)
    dataset.sort_by_length()
    dataset.shuffle_indices()
    if val_dataset_idx:
        val_dataset = asr.dataset.Audio.from_csv(
            val_dataset_idx,
            batch_size=batch_size, use_filesizes=True)

    #opt_instance = tf.optimizers.Adam(0.001 * hvd.size(), beta_1=0.9, beta_2=0.999)
    opt = tfa.optimizers.NovoGrad(0.05, beta_1=0.95,
                                  beta_2=0.5, weight_decay=0.001)

    pipeline = get_pipeline(dist_model, opt)

    callbacks = [
    ]
    schedule = tf.keras.experimental.CosineDecayRestarts(
        0.05, 10, t_mul=2.0, m_mul=0.7, alpha=0.0
    )
    callbacks.append(LearningRateScheduler(schedule))

    prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
    monitor_metric_name = 'loss' if not val_dataset_idx else 'val_loss'  # val_loss is wrong and broken
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, prefix + '_best.ckpt'),
            monitor=monitor_metric_name, save_weights_only=True,
            save_best_only=True))
    if tensorboard:
        logdir = os.path.join(model_dir, 'tb', prefix)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        callbacks.append(tensorboard_callback)

    time_start = time.time()

    hist = pipeline.fit(dataset, epochs=epochs, dev_dataset=val_dataset,
                        #steps_per_epoch=270,
                        callbacks=callbacks,
                        verbose=1,
                        #workers=2, use_multiprocessing=True,
    )
    elapsed = time.time() - time_start

    print(f'Elapsed time: {elapsed}')
    #np.save(os.path.join(model_dir, prefix + '_hist.p'), np.array(hist))


train_model(
    filename='./models/quartznet_5x5_mp.h5',
    dataset_idx='./data/train-clean-100-index.csv',
    n_blocks=1,
    is_mixed_precision=True,
    val_dataset_idx='./data/dev-clean-index.csv',
    batch_size=64,
    epochs=1000,
    tensorboard=True,
    restart_filename=None,
)
