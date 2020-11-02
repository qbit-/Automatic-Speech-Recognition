import abc
import itertools
from typing import List
import math
import numpy as np
import tensorflow as tf


def is_prefix(sequence, prefix):
    if len(sequence) <= len(prefix):
        return False
    for element1, element2 in zip(sequence, prefix):
        if element1 != element2:
            return False
    return True


def filter_dict(original, f):
    new_dict = {}
    
    for key, val in original.items():
        if f(key, val):
            new_dict[key] = val
            
    return new_dict

def get_layer_by_prefix(model, prefix):
    for layer in model.layers:
        if layer.name.startswith(prefix):
            return layer
            

class RNNTGreedyDecoder:
    """
    Use greedy search from https://arxiv.org/pdf/1211.3711.pdf to decode rnnt output.
    """
    
    def __init__(self, model, blank_label=0):
        self.blank_label = blank_label
                
        self.pred_network = get_layer_by_prefix(model, 'prediction_network')
        self.label_num = self.pred_network.input_shape[-1]
        self.encoder_network = get_layer_by_prefix(model, 'encoder_network')
        self.joint_network = get_layer_by_prefix(model, 'joint_network')
    
    def __call__(self, X):
        current_sequence = tuple()
        encoder_output = self.encoder_network(X[np.newaxis, :, :])[0][0]
        
        pred_output, state = self.call_pred(None, None)
        for t in range(len(encoder_output)):
            while self.get_next(encoder_output[t], pred_output) != self.blank_label:
                new_symbol = self.get_next(encoder_output[t], pred_output)
                current_sequence = current_sequence + (new_symbol,)
                pred_output, state = self.call_pred(new_symbol, state)
            
        return current_sequence
        
    def call_pred(self, new_symbol, state=None):
        network_input = np.zeros([1, 1, self.label_num])
        if new_symbol is not None:
            network_input[0, 0, :] = tf.one_hot(new_symbol, self.label_num)
        
        output, state = self.pred_network(network_input, states=state)
        return output[0, 0, :], state
    
    def get_transitions(self, encoder_output, pred_output):
        joint_output = self.joint_network([encoder_output[None, None, :], 
                                           pred_output[None, None, :]])
        return tf.nn.softmax(joint_output)[0, 0, 0, :]
    
    def get_next(self, encoder_output, pred_output):
        return np.argmax(self.get_transitions(encoder_output, pred_output))