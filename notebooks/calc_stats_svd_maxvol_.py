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
                                                remove_dropouts,
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

def find_activations(model, dataset_idx, size_limit=None, verbose=False):
    dataset = asr.dataset.Audio.from_csv(dataset_idx, batch_size=1, use_filesizes=True, librosa_read=False)
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

def svd_activations(model, dataset_idx, num_rows=None, verbose=False, return_activations=False, **kwargs):
    activations = find_activations(model, dataset_idx, size_limit=num_rows, verbose=verbose)
    
    if return_activations:
        return np.linalg.svd(activations, **kwargs), activations
    else:
        return np.linalg.svd(activations, **kwargs)
    
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

def get_maxvolled_deepspeech(deepspeech_model, rank, tol, change_layers=None, maxvol_type='2vol'):
    assert maxvol_type in ('2vol', '1vol')
    if change_layers is None:
        change_layers = ['dense_2', 'dense_3', 'dense_4', 'lstm_1']
    
    # clone model and rename layers to avoid problems arising from graph manipulations
    deepspeech_ = get_renamed_model(deepspeech_model)
    new_model_layers = []
    skip_next_layer = False
    for i, layer in enumerate(deepspeech_.layers):
        if skip_next_layer: 
            skip_next_layer = False
            continue
            
        if deepspeech_model.layers[i].name == 'lstm_1' and 'lstm_1' in change_layers:
            compact_v = np.load(f'./layer_activations/vt_lstm_1.npy')[:rank].T
            maxvol_idxs = np.load(f'./layer_activations/{maxvol_type}_tol{tol}_maxvolrows_lstm_1_r{rank}.npy')
            new_model_layers.extend(maxvol.get_maxvol_decomposed_lstm(layer, compact_v, maxvol_idxs))
        elif deepspeech_model.layers[i].name in change_layers:
            assert isinstance(layer, keras.layers.Dense), f"Specified layer should be dense and it is {deepspeech_.layers[i]}"
            assert isinstance(deepspeech_.layers[i + 1], keras.layers.ReLU), f"Specified dense should be followed by ReLu instead is {deepspeech_.layers[i+1]}"
            relu_name = deepspeech_model.layers[i + 1].name
            
            compact_v = np.load(f'./layer_activations/vt_{relu_name}.npy')[:rank].T
            maxvol_idxs = np.load(f'./layer_activations/{maxvol_type}_tol{tol}_maxvolrows_{relu_name}_r{rank}.npy')
            new_model_layers.extend(maxvol.get_maxvol_decomposed_dense(dense_layer=layer, 
                                                                       activation=deepspeech_model.layers[i + 1], 
                                                                       compact_v=compact_v,
                                                                       maxvol_idxs=maxvol_idxs))        
            skip_next_layer = True
        else:
            new_model_layers.append(layer)
        
    result_model = keras.Sequential()
    result_model.add(keras.Input(shape=deepspeech_model.input_shape[1:]))
    for layer in new_model_layers:
        result_model.add(layer)
    result_model = remove_dropouts(result_model)
    result_model = merge_neighbor_layers(result_model)
    result_model = get_renamed_model(result_model)
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

def get_maxvol_stats(ranks, deepspeech_model, change_layers=None, maxvol_type='2vol', tol=1.01, verbose=False):
    assert maxvol_type in ('2vol', '1vol')
    losses = []
    wers = []
    cers = []
    for rank in ranks:
        result_model = get_maxvolled_deepspeech(deepspeech_model, rank, tol, change_layers, maxvol_type)
        result_model.save(f'./models/{_layers_to_name(change_layers)}_{maxvol_type}_tol{tol}_maxvolled_deepspeech_r{rank}.h5')
#         print(result_model.get_weights())
#         result_model = keras.models.load_model(f'./models/{_layers_to_name(change_layers)}_{maxvol_type}_tol{tol}_maxvolled_deepspeech_r{rank}.h5')
        result_model.call = tf.function(result_model.call, experimental_relax_shapes=True)
        if verbose:
            print(f"After maxvol with rank {rank} and tol {tol} got model summary:")
            result_model.summary()
        loss, wer, cer = evaluate(result_model)
        
        losses.append(loss)
        wers.append(wer)
        cers.append(cer)
    return losses, wers, cers

ranks = list(reversed([100, 300, 500, 700, 1000, 1200, 1400, 1500, 1700, 1900]))
for tol in [1.01, 1.05, 1.1]:
    for maxvol_type in ('2vol', '1vol'):
        for change_layer in ('lstm_1', 'dense_2', 'dense_3', 'dense_4'):
            if os.path.exists(f'./maxvol_stats/{change_layer}_{maxvol_type}_tol{tol}_wers.npy'):
                print(f"Skip {maxvol_type} {change_layer} layers")
                continue
            else:
                np.save(f'./maxvol_stats/{change_layer}_{maxvol_type}_tol{tol}_cers.npy', np.array([0,0,0,0]))

            losses, wers, cers = get_maxvol_stats(ranks, deepspeech, 
                                                  change_layers=[change_layer], 
                                                  maxvol_type=maxvol_type, 
                                                  tol=tol,
                                                  verbose=True)
            np.save(f'./maxvol_stats/{change_layer}_{maxvol_type}_tol{tol}_wers.npy', np.array(wers))
            np.save(f'./maxvol_stats/{change_layer}_{maxvol_type}_tol{tol}_cers.npy', np.array(cers))
            np.save(f'./maxvol_stats/{change_layer}_{maxvol_type}_tol{tol}_losses.npy', np.array(losses))
            import sys
            sys.exit(0)
        


for tol in [1.01, 1.05, 1.1]:
    for maxvol_type in ('2vol', '1vol'):
        if os.path.exists(f'./maxvol_stats/{maxvol_type}_tol{tol}_cers.npy'):
            print(f"Skip {maxvol_type} all layers")
            continue
        else:
            np.save(f'./maxvol_stats/{maxvol_type}_tol{tol}_cers.npy', np.array([0,0,0,0]))
        losses, wers, cers = get_maxvol_stats(ranks, deepspeech, 
                                              change_layers=('lstm_1', 'dense_2', 'dense_3', 'dense_4'), 
                                              tol=tol,
                                              maxvol_type=maxvol_type, 
                                              verbose=True)
        np.save(f'./maxvol_stats/{maxvol_type}_tol{tol}_wers.npy', np.array(wers))
        np.save(f'./maxvol_stats/{maxvol_type}_tol{tol}_cers.npy', np.array(cers))
        np.save(f'./maxvol_stats/{maxvol_type}_tol{tol}_losses.npy', np.array(losses))
        import sys
        sys.exit(0)