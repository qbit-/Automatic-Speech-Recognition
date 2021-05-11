# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: asr
#     language: python
#     name: asr
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
import requests
import matplotlib.pyplot as plt
from zipfile import ZipFile
from tarfile import TarFile

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


def download(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            chunk_size = max(int(int(total)/100), 1024*1024)
            total_chunks = int(int(total) / chunk_size)
            
            for data in tqdm(response.iter_content(chunk_size=chunk_size), total=total_chunks):
                downloaded += len(data)
                f.write(data)


def extract_nemo_files(zip_filename, destination, keep_files=False):
    os.makedirs(destination, exist_ok=True)
    with ZipFile(zip_filename, 'r') as zfp:
        zfp.extractall(destination)
    nemo_filename = os.path.join(destination, 'JasperNet10x5-En-Base.nemo')
    tar_filename = nemo_filename.rsplit('.')[0] + '.tar'
    
    os.rename(nemo_filename, tar_filename)
    with TarFile.open(tar_filename, 'r:gz') as zfp:
        for target in ['JasperDecoderForCTC.pt', 'JasperEncoder.pt']:
            for tarinfo in zfp.getmembers():
                if tarinfo.name.endswith(target):
                    tarinfo.name = target
                    zfp.extract(tarinfo, destination)
                    break
    
    if not keep_files:
        os.remove(zip_filename)
        os.remove(tar_filename)


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

# ### Basic test

model = get_jasperdr(input_dim=64, output_dim=29,
                     is_mixed_precision=False,
                     fixed_sequence_size=None,
                     num_b_block_repeats=2,
                     b_block_kernel_sizes=(11, 13, 17, 21, 25),
                     b_block_num_channels=(256, 384, 512, 640, 768),
                     num_small_blocks=5,
                     use_biases=False,
                     use_batchnorms=True,
                     fixed_batch_size=None,
                     random_state=1)

# + jupyter={"outputs_hidden": true} tags=[]
model.input
# -

dataset = asr.dataset.Audio.from_csv('./data/libri-dev-clean-index.csv', batch_size=3)
for audio, transcripts in tqdm(dataset, position=0):
    features, _ = features_extractor(audio)
    x = model.predict(features)
    decoded_labels = decoder(x)
    predictions = alphabet.get_batch_transcripts(decoded_labels)
    print(predictions)
    break

# ### Test with pretrained weights

download('https://api.ngc.nvidia.com/v2/models/nvidia/multidataset_jasper10x5dr/versions/5/zip',
         'data/multidataset_jasper10x5dr_5.zip')

download('https://api.ngc.nvidia.com/v2/models/nvidia/jaspernet10x5dr/versions/1/zip',
         'data/jaspernet10x5dr_1.zip')



extract_nemo_files('data/multidataset_jasper10x5dr_5.zip', 'data/jaspernet10x5dr', keep_files=False)



model = load_nvidia_jasperdr(
    './data/jaspernet10x5dr/JasperEncoder_3-STEP-218410.pt',
    './data/jaspernet10x5dr/JasperDecoderForCTC_4-STEP-218410.pt',
    fixed_sequence_size=None,
    fixed_batch_size=4)

dataset = asr.dataset.Audio.from_csv('./data/libri-test-clean-index.csv', batch_size=3)
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
    './data/jaspernet10x5dr/JasperEncoder.pt',
    './data/jaspernet10x5dr/JasperDecoderForCTC.pt')
    pipeline = asr.pipeline.CTCPipeline(
    alphabet, features_extractor, model, optimizer, decoder
)
evaluate_model(model, pipeline, './data/libri-dev-clean-index.csv', batch_size=1)

# %%time
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")
model = load_nvidia_jasperdr(
    './data/jaspernet10x5dr/JasperEncoder.pt',
    './data/jaspernet10x5dr/JasperDecoderForCTC.pt')
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
export_model(model, 'data/jaspernet10x5dr/jasper_dr_10x5_test.tflite', custom_objects=custom_objects)








