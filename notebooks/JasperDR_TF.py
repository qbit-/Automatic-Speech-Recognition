# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: conda_asr_project
#     language: python
#     name: conda_asr_ptoject
# ---

# %load_ext autoreload
# %autoreload 2

# +
import warnings

import os
import sys
if os.path.abspath('../') not in sys.path:
    sys.path.append(os.path.abspath('../'))
if os.path.abspath('../../tt_keras') not in sys.path:
    sys.path.append(os.path.abspath('../../tt_keras'))

import numpy as np
import tensorflow as tf
import math
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

from tensorflow.keras.layers import BatchNormalization, Conv1D, SeparableConv1D
from tensorflow.python.keras.layers.convolutional import SeparableConv
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import activations
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec

import automatic_speech_recognition as asr
from automatic_speech_recognition.model.jasperdr import get_jasperdr, load_nvidia_jasperdr
from automatic_speech_recognition.model.jasperdr import B_block, Small_block
from automatic_speech_recognition.evaluate.evaluate import get_metrics
from converter_utils import (export_model, clone_and_fix_shape)


# -

def evaluate_model(model, pipeline, index_file, batch_size=1):
    dataset = asr.dataset.Audio.from_csv(index_file, batch_size=batch_size)
    wer, cer = asr.evaluate.calculate_error_rates(pipeline, dataset)
    
    return {'wer': wer, 'cer': cer}


alphabet = asr.text.Alphabet(lang='en')
features_extractor = asr.features.FilterBanks(
    features_num=64,
    sample_rate=16000,
    winlen=0.02,
    winstep=0.01,
    window="hann",
)
optimizer = tf.optimizers.Adam(
    lr=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)
decoder = asr.decoder.GreedyDecoder()

model = get_jasperdr(input_dim=64, output_dim=29,
                      is_mixed_precision=False,
                      tflite_version=False,
                      num_b_block_repeats=2,
                      b_block_kernel_sizes=(11, 13, 17, 21, 25),
                      b_block_num_channels=(256, 384, 512, 640, 768),
                      num_small_blocks=5,
                      use_biases=False,
                      use_batchnorms=True,
                      fixed_batch_size=10,
                      random_state=1)

model.input

dataset = asr.dataset.Audio.from_csv('./data/libri-dev-clean-index.csv', batch_size=3)
for audio, transcripts in tqdm(dataset, position=0):
    features, _ = features_extractor(audio)
    x = model.predict(features)
    decoded_labels = decoder(x)
    predictions = alphabet.get_batch_transcripts(decoded_labels)
    print(predictions)
    break

model = load_nvidia_jasperdr(
    './data/jaspernet10x5dr/JasperEncoder_3-STEP-218410.pt',
    './data/jaspernet10x5dr/JasperDecoderForCTC_4-STEP-218410.pt',
    tflite_version=True,
    fixed_batch_size=4)

dataset = asr.dataset.Audio.from_csv('./data/libri-small-clean-index.csv', batch_size=3)
for audio, transcripts in tqdm(dataset, position=0):
    features, _ = features_extractor(audio)
    x = model.predict(features)
    decoded_labels = decoder(x)
    predictions = alphabet.get_batch_transcripts(decoded_labels)
    print(predictions)
    break

# %%time
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")
model = load_nvidia_jasperdr(
    './data/jaspernet10x5dr/JasperEncoder_3-STEP-218410.pt',
    './data/jaspernet10x5dr/JasperDecoderForCTC_4-STEP-218410.pt')
pipeline = asr.pipeline.CTCPipeline(
    alphabet, features_extractor, model, optimizer, decoder
)
evaluate_model(model, pipeline, './data/libri-dev-clean-index.csv', batch_size=1)

# %%time
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")
model = load_nvidia_jasperdr(
    './data/jaspernet10x5dr/JasperEncoder_3-STEP-218410.pt',
    './data/jaspernet10x5dr/JasperDecoderForCTC_4-STEP-218410.pt')
pipeline = asr.pipeline.CTCPipeline(
    alphabet, features_extractor, model, optimizer, decoder
)
evaluate_model(model, pipeline, './data/libri-test-clean-index.csv', batch_size=1)

# + active=""
#

# +
import os, sys
if os.path.abspath('../tt_keras') not in sys.path:
    sys.path.append(os.path.abspath('../tt_keras'))
    
from converter_utils import export_model, clone_and_fix_shape
from device_profiling import (check_device, run_on_device, parse_profiler_output, batch_profile)
from device_profiling import DEFAULT_PROF_CONFIG as config
# -

# %%time
custom_objects = {"B_block": B_block, "Small_block": Small_block}
export_model(model, 'models/jasper/jasper_dr_10x5_test.tflite', custom_objects=custom_objects)


