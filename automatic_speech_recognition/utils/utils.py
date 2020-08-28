import os
import pickle
import logging
from functools import reduce
from logging import Logger
from typing import Any, Tuple
import numpy as np
import librosa
from tensorflow import keras
from google.cloud import storage
import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
import time

logger = logging.getLogger('asr.utils')


def load(file_path: str):
    """ Load arbitrary python objects from the pickled file. """
    with open(file_path, mode='rb') as file:
        return pickle.load(file)


def save(data: Any, file_path: str):
    """ Save arbitrary python objects in the pickled file. """
    with open(file_path, mode='wb') as file:
        pickle.dump(data, file)


def download_from_bucket(bucket_name: str, remote_path: str, local_path: str):
    """ Download the file from the public bucket. """
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = storage.Blob(remote_path, bucket)
    blob.download_to_filename(local_path, client=client)


def maybe_download_from_bucket(bucket_name: str, remote_path: str, local_path: str):
    """ Download file from the bucket if it does not exist. """
    if os.path.isfile(local_path):
        return
    directory = os.path.dirname(local_path)
    os.makedirs(directory, exist_ok=True)
    logger.info('Downloading file from the bucket...')
    download_from_bucket(bucket_name, remote_path, local_path)


def read_audio(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read audiofile to np.ndarray of np.float32 numbers,
    normalized in between [-1, 1]
    """
    audio, sample_rate = librosa.load(file_path, sr=None)
    return audio, sample_rate

def tf_read_audio(file_path: str):
    """
    Read audiofile to np.ndarray of np.float32 numbers,
    normalized in between [-1, 1]
    """
    audio = tf.io.read_file(file_path)
    waveform = tf.audio.decode_wav(audio)
    return waveform, 16000


def calculate_units(model: keras.Model) -> int:
    """ Calculate number of the model parameters. """
    units = 0
    for parameters in model.get_weights():
        units += reduce(lambda x, y: x * y, parameters.shape)
    return units


def create_logger(file_path=None, level=20, name='asr') -> Logger:
    """ Create the logger and handlers both console and file. """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] [%(name)-20s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)  # handle all messages from logger
    if file_path:
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def profile(function_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Function {function_name} call invoked")
            start_time = time.time()
            res = func(*args, **kwargs)
            print(f"Fucntion {function_name} call ended "
                  f"in {time.time() - start_time}")
            return res
        return wrapper
    return decorator


def get_renamed_model(source_model):
    assert isinstance(source_model, keras.Sequential)
    
    def clone_func_reset_name(layer, **kwargs):
        original_config = layer.get_config()
        del original_config['name']
        return layer.__class__.from_config(original_config)
    
    new_model = tf.keras.Sequential()
    new_model.add(keras.Input(shape=source_model.input_shape[1:]))
    for i, layer in enumerate(source_model.layers):
        new_model.add(clone_func_reset_name(layer))
    new_model.set_weights(source_model.get_weights())
    
    return new_model


def load_graph_from_gfile(gfile_path, verbose=False):
    """
    Loads graph from a ForzenGraph (.pb) file and extracts values of
    tensors
    """
    gr = tf.Graph()
    wts = {}
    with tf.compat.v1.Session(graph=gr) as sess:
        print("load graph")
        with gfile.FastGFile(gfile_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='', )
        graph_nodes = [n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            names.append(t.name)
        if verbose:
            print(names)
        all_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        if verbose:
            print(all_vars)
        for n in graph_nodes:
            if n.op == "Const":
                wts[n.name] = tensor_util.MakeNdarray(
                    n.attr["value"].tensor)
    return wts, gr


def profile(function_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Function {function_name} call invoked")
            start_time = time.time()
            res = func(*args, **kwargs)
            print(f"Fucntion {function_name} call ended "
                  f"in {time.time() - start_time}")
            return res
        return wrapper
    return decorator


def select_layers(model, names=None, trainable_only=False, non_trainable_only=False, other_predicate=None):
    """
    Returns array of keras.Layer objects from model with specified names which satisfy other conditions.
    If names is None then all layers are considered.
    """
    layers = list(model.layers)
    if names is not None:
        layers = filter(lambda x: x.name in names, layers)
    if trainable_only:
        layers = filter(lambda x: len(x.trainable_variables) > 0, layers)
    if non_trainable_only:
        layers = filter(lambda x: len(x.trainable_variables) == 0, layers)
    if other_predicate is not None:
        layers = filter(other_predicate, layers)
    return list(layers)


def wrap_call_methods(model, wrapper, names=None, trainable_only=False, non_trainable_only=False, 
                      wrap_rnn_cells=False, wrap_time_distributed_inner=False):
    """
    Wraps call function of specified layers in wrapper decorator.
    
    model: model which layers will wrapped. Argument is mutated.
    wrapper: callable which recives layer and 
        returns wrapped call method.
    layers: list of layer names to be wrapped. If None then all
        layers will be wrapped.
    trainable_only: wrap operation will only be applied to layers 
        with trainable parameters.
    non_trainable_only: wrap operation will only be applied to layers 
        without trainable parameters.
    wrap_rnn_cells: if True than instead of RNN layer rnn cell 
        call will wraped.
    wrap_time_distributed_inner: if True than instead of TimeDistruted layer
        its inner layer will be wrapped
    """
    for layer in select_layers(model, names, trainable_only, non_trainable_only):
        if wrap_rnn_cells and isinstance(layer, keras.layers.RNN):
            layer.cell.call = wrapper(layer.cell)
        elif wrap_time_distributed_inner and isinstance(layer, keras.layers.TimeDistributed):
            layer.layer.call = wrapper(layer.layer)
        else:
            layer.call = wrapper(layer)
    return model
