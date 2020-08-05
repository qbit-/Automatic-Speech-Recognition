import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
import tensorflow_addons as tfa
import sys
import os
if os.path.abspath('../') not in sys.path:
    sys.path.append(os.path.abspath('../'))
if os.path.abspath('../../tt_keras') not in sys.path:
    sys.path.append(os.path.abspath('../../tt_keras'))
if os.path.abspath('../../tf2-gradient-checkpointing') not in sys.path:
    sys.path.append(os.path.abspath('../../tf2-gradient-checkpointing'))

if os.path.abspath('../../t3f') not in sys.path:
    sys.path.append(os.path.abspath('../../t3f'))

import automatic_speech_recognition as asr
from automatic_speech_recognition.utils import (wrap_call_methods, select_layers, 
                                                apply_dense, remove_dropouts, apply_lstm,
                                                merge_dense_dense, merge_dense_lstm,
                                                merge_neighbor_layers, get_renamed_model,
                                                get_model_prefixes)
from automatic_speech_recognition.dataset import ModelOutputsDataset
from automatic_speech_recognition.model import maxvol
import time
from datetime import datetime
import argparse
import pickle
from checkpointing import checkpointable
from functools import partial
from transform_model import transform
from tqdm import tqdm
from tensorflow.keras import layers
import seaborn as sns
sns.set()

from collections.abc import Iterable, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm.notebook import tqdm
from h5_to_tflite import TF_CUSTOM_OBJECTS

from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import horovod.tensorflow.keras as hvd

#%load_ext tensorboard
#%tensorboard --logdir=./models/ --port=32779

# !!!ONLY RUN ONCE AT THE START
deepspeech = asr.model.load_mozilla_deepspeech('./data/myfrozen.pb', verbose=False)
deepspeech.summary()

def get_pipeline(model, optimizer=None):
    alphabet = asr.text.Alphabet(lang='en')
    features_extractor = asr.features.TfMFCC(
        features_num=26,
        winlen=0.032,
        winstep=0.02,
    )
    if not optimizer:
        optimizer = tf.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
    decoder = asr.decoder.GreedyDecoder()
    pipeline = asr.pipeline.CTCPipeline(
        alphabet, features_extractor, model, optimizer, decoder
    )
    return pipeline

def get_optimizer():
    opt_instance = tfa.optimizers.NovoGrad(0.0, beta_1=0.95, beta_2=0.5, weight_decay=0.001)
    return hvd.DistributedOptimizer(opt_instance)

def find_activations(model, dataset_idx, batch_size=12, size_limit=None, verbose=False):
    dataset = asr.dataset.Audio.from_csv('./data/dev-clean-index.csv', batch_size=10, use_filesizes=True, librosa_read=False)
    dataset.sort_by_length()
    dataset.shuffle_indices()
    dataset = ModelOutputsDataset(get_pipeline(model), dataset)

    activations = []
    num_activations = 0
    for i, (_, output) in enumerate(dataset):
        output = (output
                  .cpu()
                  .numpy()
                  .reshape(-1, output.shape[-1]))
        activations.append(output)
        num_activations += len(activations[-1])    
        if size_limit is not None and num_activations >= size_limit:
            break
        if verbose and i % 50 == 0:
            print(f"Calculated batch {i} out of {len(dataset)}")
    
    activations = np.concatenate(activations)
    if size_limit is not None:
        activations = activations[:size_limit]
        
    return activations    
    
def evaluate(model, dataset_idx='./data/dev-clean-index.csv'):
    pipeline = get_pipeline(model)
    dataset = asr.dataset.Audio.from_csv(dataset_idx, batch_size=10, use_filesizes=True, librosa_read=False)
    dataset.sort_by_length()
    pipeline.compile_model()

    dataset = pipeline.wrap_preprocess(dataset)
    loss = model.evaluate(dataset)
    
    test_dataset = asr.dataset.Audio.from_csv(dataset_idx, batch_size=10, use_filesizes=True, librosa_read=False)
    dataset.sort_by_length()
    wer, cer = asr.evaluate.calculate_error_rates(pipeline, test_dataset)
    return loss, wer, cer


def get_maxvolled_deepspeech(deepspeech_model, rank, change_layers=None, maxvol_type='2vol'):
    assert maxvol_type in ('2vol', '1vol')
    if change_layers is None:
        change_layers = [ 'dense_2', 'dense_3', 'dense_4', 'lstm_1']
    
    # clone model and rename layers to avoid problems arising from graph manipulations
    deepspeech_ = get_renamed_model(deepspeech_model)
    # get input of a model. Without fixing, InputLayers break process of iterating over model
    layers = deepspeech_.layers
    start_layer = 0
    if isinstance(layers[0], keras.layers.InputLayer):
        output = layers[0].output
        start_layer = 1
    else:
        output = deepspeech_.input    

    for i in range(start_layer, len(layers)):
        if deepspeech_model.layers[i].name == 'lstm_1' and 'lstm_1' in change_layers:
            compact_v = np.load(f'./layer_activations/vt_lstm_1.npy')[:rank].T
            maxvol_idxs = np.load(f'./layer_activations/{maxvol_type}_maxvolrows_lstm_1_r{rank}.npy')
            output = maxvol.apply_maxvol_decomposed_lstm(output, layers[i], compact_v, maxvol_idxs)
        elif i + 1 < len(layers) and deepspeech_model.layers[i].name in change_layers:
            assert isinstance(layers[i + 1], keras.layers.ReLU), f"Specified layers should ReLu layers instead it is {layers[i+1]}"
            assert isinstance(layers[i], keras.layers.Dense), f"Layer before specified ReLu should be Dense and it is {layers[i]}"
            relu_name = deepspeech_model.layers[i + 1].name
            
            compact_v = np.load(f'./layer_activations/vt_{relu_name}.npy')[:rank].T
            maxvol_idxs = np.load(f'./layer_activations/{maxvol_type}_maxvolrows_{relu_name}_r{rank}.npy')
            output = maxvol.apply_maxvol_decomposed_dense(output, 
                                                dense_layer=layers[i], 
                                                activation=layers[i + 1], 
                                                compact_v=compact_v,
                                                maxvol_idxs=maxvol_idxs)
        
        else:
            output = layers[i](output)
        
    result_model = keras.Model(deepspeech_.input, output)
    result_model = get_renamed_model(result_model)
    result_model = merge_neighbor_layers(remove_dropouts(result_model))
    
    return result_model

def _layers_to_name(layers_to_change):
    result = ''
    if 'lstm_1' in layers_to_change:
        result = 'lstm'
        
    denses = []
    for entry in layers_to_change:
        if entry.startswith('dense_'):
            denses.append(entry[len('dense_'):])
    denses = sorted(denses)
    
    if len(denses) > 0:
        if len(result) > 0:
            result += '_'
        result += 'desnse' + ''.join(denses)
        
    return result

def get_maxvol_stats(ranks, deepspeech_model, change_layers=None, maxvol_type='2vol', verbose=False):
    assert maxvol_type in ('2vol', '1vol')
    losses = []
    wers = []
    cers = []
    for rank in ranks:
        result_model = get_maxvolled_deepspeech(deepspeech_model, rank, change_layers, maxvol_type)
        result_model.save(f'./models/{_layers_to_name(change_layers)}_{maxvol_type}_maxvolled_deepspeech_r{rank}.h5')
        result_model.call = tf.function(result_model.call, experimental_relax_shapes=True)
        if verbose:
            print(f"After maxvol with rank {rank} got model summary:")
            print(result_model.summary())

        loss, wer, cer = evaluate(result_model)
        losses.append(loss)
        wers.append(wer)
        cers.append(cer)
    return losses, wers, cers


ranks = list(reversed([100, 300, 500, 700, 1000, 1200, 1400, 1500, 1700]))
for maxvol_type in ('2vol', '1vol'):
    for change_layer in ('lstm_1', 'dense_2', 'dense_3', 'dense_4'):
        if os.path.exists(f'./maxvol_stats/{change_layer}_{maxvol_type}_wers.npy'):
            continue
        losses, wers, cers = get_maxvol_stats(ranks, deepspeech, 
                                              change_layers=[change_layer], 
                                              maxvol_type=maxvol_type, 
                                              verbose=True)
        np.save(f'./maxvol_stats/{change_layer}_{maxvol_type}_wers.npy', np.array(wers))
        np.save(f'./maxvol_stats/{change_layer}_{maxvol_type}_cers.npy', np.array(cers))
        np.save(f'./maxvol_stats/{change_layer}_{maxvol_type}_losses.npy', np.array(losses))
        
for maxvol_type in ('2vol', '1vol'):
    if os.path.exists(f'./maxvol_stats/{maxvol_type}_wers.npy'):
        continue
    losses, wers, cers = get_maxvol_stats(ranks, deepspeech, 
                                          change_layers=('lstm_1', 'dense_2', 'dense_3', 'dense_4'), 
                                          maxvol_type=maxvol_type, 
                                          verbose=True)
    np.save(f'./maxvol_stats/{maxvol_type}_wers.npy', np.array(wers))
    np.save(f'./maxvol_stats/{maxvol_type}_cers.npy', np.array(cers))
    np.save(f'./maxvol_stats/{maxvol_type}_losses.npy', np.array(losses))
    
def get_cropped_svd_deepspeech(deepspeech_model, rank, change_layers=None):
    if change_layers is None:
        change_layers = [ 'dense_2', 'dense_3', 'dense_4', 'lstm_1']
    
    # clone model and rename layers to avoid problems arising from graph manipulations
    deepspeech_ = get_renamed_model(deepspeech_model)
    # get input of a model. Without fixing, InputLayers break process of iterating over model
    layers = deepspeech_.layers
    start_layer = 0
    if isinstance(layers[0], keras.layers.InputLayer):
        output = layers[0].output
        start_layer = 1
    else:
        output = deepspeech_.input    
    
    for i in range(start_layer, len(layers)):
        if (deepspeech_model.layers[i].name == 'lstm_1' and 'lstm_1' in change_layers or
                i - 1 > 0 and deepspeech_model.layers[i - 1].name in change_layers and isinstance(layers[i], keras.layers.ReLU)):
            output = layers[i](output)
            compact_v = np.load(f'./layer_activations/vt_lstm_1.npy')[:rank].T
            output = apply_dense(output, compact_v)
            output = apply_dense(output, np.linalg.pinv(compact_v))
        else:
            output = layers[i](output)
    
    result_model = keras.Model(deepspeech_.input, output)
    result_model = get_renamed_model(result_model)
    result_model = merge_neighbor_layers(remove_dropouts(result_model))
    return result_model

def get_cropped_svd_stats(ranks, deepspeech_model, change_layers=None, verbose=False):
    losses = []
    wers = []
    cers = []
    for rank in ranks:
        # clone model and rename layers to avoid problems arising from graph manipulations  
        result_model = get_cropped_svd_deepspeech(deepspeech_model, rank, change_layers)
        result_model.call = tf.function(result_model.call, experimental_relax_shapes=True)
        if verbose:
            print(f"After cropped svd with rank {rank} got model summary:")
            print(result_model.summary())

        loss, wer, cer = evaluate(result_model) 
        losses.append(loss)
        wers.append(wer)
        cers.append(cer)
    return losses, wers, cers

ranks = list(reversed([100, 300, 500, 700, 1000, 1200, 1400, 1500, 1700]))

if not os.path.exists(f'./cropsvd_stats/wers.npy'):
    losses, wers, cers = get_cropped_svd_stats(ranks, deepspeech, 
                                          change_layers=('lstm_1', 'dense_2', 'dense_3', 'dense_4'), 
                                          verbose=True)

    np.save(f'./cropsvd_stats/wers.npy', np.array(wers))
    np.save(f'./cropsvd_stats/cers.npy', np.array(cers))
    np.save(f'./cropsvd_stats/losses.npy', np.array(losses))

for change_layer in ('lstm_1', 'dense_2', 'dense_3', 'dense_4'):
    if os.path.exists(f'./cropsvd_stats/{change_layer}_wers.npy'):
        continue
    losses, wers, cers = get_cropped_svd_stats(ranks, deepspeech, 
                                          change_layers=[change_layer], 
                                          verbose=True)
    np.save(f'./cropsvd_stats/{change_layer}_wers.npy', np.array(wers))
    np.save(f'./cropsvd_stats/{change_layer}_cers.npy', np.array(cers))
    np.save(f'./cropsvd_stats/{change_layer}_losses.npy', np.array(losses))