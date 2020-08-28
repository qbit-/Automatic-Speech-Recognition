from collections import deque
import tensorflow as tf
from tensorflow import keras
import numpy as np

def remove_dropouts(model):
    """
    Returns model without dropout layers. 
    ! ONLY FOR BRANCHLESS FUNCTIONAL API MODELS
    """
    new_model = keras.Sequential()
    new_model.add(keras.Input(shape=model.input_shape[1:]))
    for layer in model.layers:
        if not isinstance(layer, keras.layers.Dropout):
            new_model.add(layer)
    
    return new_model


def get_dense(weights, biases=None):
    """
    Creates Dense layer with specified weights 
    Returns output tensor.
    !ONLY FOR FUNCTIONAL API
    """
    if biases is None:
        biases = np.zeros(weights.shape[1])
    assert weights.shape[1] == biases.shape[0], "biases and weights shape should be equal"
    layer = keras.layers.Dense(weights.shape[1])
    layer.build((None, weights.shape[0]))
    layer.set_weights([weights, biases])
    
    return layer


def get_conv(weights, biases=None):
    """
    Creates Dense layer with specified weights 
    Returns output tensor.
    !ONLY FOR FUNCTIONAL API
    """
    if biases is None:
        biases = np.zeros(weights.shape[1])
    assert weights.shape[3] == biases.shape[0], "biases and weights shape should be equal"
    layer = keras.layers.Conv2D(weights.shape[1])
    layer.build((None, weights.shape[0]))
    layer.set_weights([weights, biases])
    
    return layer

def get_lstm(W1, W2, biases=None, **kwargs):
    """
    Creates LSTM layer with specified weights 
    Returns output tensor.
    !ONLY FOR FUNCTIONAL API
    """
    if biases is None:
        biases = np.zeros(W1.shape[1])
    assert W2.shape[0] == W1.shape[1] // 4, f"Input shape of recurrent kernel W2 {W2.shape} must be equal to output shape of kernel W1 {W1.shape}"
    assert W1.shape[1] == biases.shape[0], "biases and W1 shape should be compatible"
    assert W2.shape[1] == biases.shape[0], "biases and W2 shape should be compatible"
    
    layer = keras.layers.LSTM(W1.shape[1] // 4, **kwargs)
    layer.build((None, None, W1.shape[0]))
    layer.set_weights([W1, W2, biases])

    return layer


def merge_dense_dense(dense_first, dense_second):
    """
    Returns Dense layer which is equivalent to applying first dense and then second dense.
    """
    W1, b1 = dense_first.get_weights()
    W2, b2 = dense_second.get_weights()
    
    # create new layer with input shape as first layer ans 
    # output shape as second layer
    new_dense = keras.layers.Dense(W2.shape[1])
    new_dense.build(W1.shape[0])
    # set weights to a composition of both linear layers
    new_dense.set_weights([W1.dot(W2), b1.dot(W2) + b2])
    
    return new_dense

def merge_conv_dense(conv, dense):
    """
    Returns Dense layer which is equivalent to applying first dense and then second dense.
    """
    W_conv, b_conv = conv.get_weights()
    W_dense, b_dense = dense.get_weights()
    
    W_conv_reshape = W_conv.reshape((-1, W_conv.shape[-1]))
    assert W_conv_reshape.shape[-1] == W_dense[0], f"Layers have incompatible shapes, conv: {W_conv.shape}, dense: {W_dense.shape}"
    
    # create new layer with input shape as first layer
    # output shape as second layer
    new_conv = keras.layers.Conv2D(W_dense.shape[1], W_conv.shape[0:2])
    new_conv.build((None, W_conv.shape[0], W_conv.shape[1], W_conv.shape[2]))
    
    new_conv_weight_reshape = W_conv_reshape @ W_dense
    new_conv_weight = new_conv_weigth.reshape(W_conv.shape[0:2] + (W_dense[1],))
    new_conv_bias = b_conv @ W_dense + b_dense
    # set weights to a composition of both linear layers
    new_conv.set_weights([new_conv_weight, new_conv_bias])
    
    return new_сщтм


def merge_dense_lstm(dense, lstm):
    """
    Returns LSTM layer which is equivalent to applying dense layer and then lstm layer.
    """
    W, b_dense = dense.get_weights()
    W1, W2, b_lstm = lstm.get_weights()
    
    # create new layer with input shape as first layer ans 
    # output shape as second layer
    new_lstm = keras.layers.LSTM(W1.shape[1] // 4, return_sequences=lstm.get_config()['return_sequences'])
    new_lstm.build((None, None, W.shape[0]))
    # set weights to a composition of both linear layers
    new_lstm.set_weights([W.dot(W1), W2, b_dense.dot(W1) + b_lstm])
    
    return new_lstm

def merge_neighbor_layers(model):
    """
    ! ONLY FOR BRANCHLESS FUNCTIONAL API MODELS.
    Returns model where pairs of neighbors dense->dense and dense->lstm are merged into one layer.
    """  
    applied_layers = deque()
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense) and isinstance(applied_layers[-1], keras.layers.Dense):
            prev_layer = applied_layers.pop()
            new_merged_dense = merge_dense_dense(prev_layer, layer)
            applied_layers.append(new_merged_dense)
        elif isinstance(layer, keras.layers.LSTM) and isinstance(applied_layers[-1], keras.layers.Dense):
            prev_layer = applied_layers.pop()
            new_merged_lstm = merge_dense_lstm(prev_layer, layer)
            applied_layers.append(new_merged_lstm)
        else:
            applied_layers.append(layer)
           
    new_model = keras.Sequential()
    new_model.add(keras.Input(shape=model.input_shape[1:]))

    for layer in applied_layers:
        new_model.add(layer)
    return new_model


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

def get_model_prefixes(model, tail_names=None, tail_numbers=None):
    """
    Get keras.Model slices of original model. Prefix can be specified by its
    output layer name or idx. 
    """
    if tail_names is None:
        assert tail_numbers is not None, "You shoul specify at least one argument to determine tails"
        tail_names = [model.layers[layer_num].name for layer_num in tail_numbers]
    
    for tail in tail_names:
        yield keras.Model(model.input, model.get_layer(tail).output)
        
def _is_rnn_input_shape(shape):
    return len(shape) == 3 and isinstance(shape[2], int)

def _change_layer_args(old_layer, input_shape=None, **kwargs):
    config = old_layer.get_config()
    for option in kwargs:
        config[option] = kwargs[option]
    new_layer = old_layer.__class__.from_config(config)
    
    if len(old_layer.get_weights()) > 0:
        if input_shape is None:
            input_shape = old_layer.input_shape
        new_layer.build(input_shape)
        new_layer.set_weights(old_layer.get_weights())
    return new_layer

def fixate_rnn_shape(model, num_steps, fixate_rnns=True):
    """
    Fixates length of input sequences to a certain number, unrolls rnns. Required for proper convertation to tflite.
    """
    assert _is_rnn_input_shape(model.input_shape), "Input layer should have proper rnn shape -- [*, *, n]. Where * is int r None"
    input_tensor = keras.Input(shape=(num_steps, model.input_shape[-1]))
    
    return keras.Model(input_tensor, model(input_tensor))