from tensorflow import keras
import tensorflow as tf
import numpy as np
from ..utils import apply_dense, apply_lstm

def apply_maxvol_decomposed_dense(input_tensor, dense_layer, activation, compact_v, maxvol_idxs):
    if isinstance(activation, str):
        activation = keras.layers.Activation(activation)
    W, b = dense_layer.get_weights()
    
    out = apply_dense(input_tensor, W[:, maxvol_idxs], b[maxvol_idxs])
    out = activation(out)
    out = apply_dense(out, np.linalg.pinv(compact_v[maxvol_idxs]).T)
    out = apply_dense(out, compact_v.T)
    return out


def get_maxvolled_lstm_weights(W1, W2, b, compact_v, maxvol_idxs):
    # W1 and W2 determine outputs of 4 dense layers iside lstm. for each we 
    # need to select maxvol_idxs outputs
    select_idxs = np.concatenate([maxvol_idxs + i * (W1.shape[1] // 4) for i in range(4)])
    # recurrent weights accept the output of the cell 
    # and they need to be modified by every dense layer which appear
    # after lstm1
    W2_new = np.linalg.pinv(compact_v[maxvol_idxs]).T @ compact_v.T @ W2[:, select_idxs]
    W1_new = W1[:, select_idxs]
    b_new = b[select_idxs]
    return W1_new, W2_new, b_new


def apply_maxvol_decomposed_lstm(input_tensor, lstm_layer, compact_v, maxvol_idxs):
    W1, W2, b = lstm_layer.get_weights()
    W1_new, W2_new, b_new = get_maxvolled_lstm_weights(W1, W2, b, compact_v, maxvol_idxs)
    out = apply_lstm(input_tensor, W1_new, W2_new, b_new, return_sequences=lstm_layer.get_config()['return_sequences'])
    out = apply_dense(out, np.linalg.pinv(compact_v[maxvol_idxs]).T)
    out = apply_dense(out, compact_v.T)
    return out
