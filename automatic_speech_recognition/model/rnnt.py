import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import copy
try:
    from warprnnt_tensorflow import rnnt_loss
except:
    logger.info("Could not import warp-rnnt loss")

def reduce_time(outputs, factor=2, time_major=True):
    """
    Taken from https://github.com/thomasschmied/Speech_Recognition_with_Tensorflow/blob/master/SpeechRecognizer.py
    Reshapes the given outputs, i.e. reduces the
    time resolution by 2.
    Similar to "Listen Attend Spell".
    https://arxiv.org/pdf/1508.01211.pdf
    """
    assert time_major
    # [max_time, batch_size, num_units]
    shape = tf.shape(outputs)
    max_time, batch_size, num_units = outputs.shape
    # if static dimension is None use runtime Tensor value
    if max_time is None:
        max_time = shape[0]
    if batch_size is None:
        batch_size = shape[1]
    if num_units is None:
        raise ValueError("Last dimension of input tensor should be known")
    
    # We need to pad s equence so that its length is divisible by reduction factor
    padding_size = tf.math.floormod(max_time - factor, factor)
    outputs = tf.pad(outputs, [[0, padding_size], [0, 0], [0, 0]])
    concat_outputs = tf.reshape(outputs, (-1, batch_size, num_units * factor))
    return concat_outputs


class TimeReductionLayer(keras.layers.Layer):
    def __init__(self, factor, time_major):
        super().__init__()
        assert time_major
        self._factor = factor
        self._time_major = time_major

    def call(self, inputs, **kwargs):
        return reduce_time(inputs, self._factor, self._time_major)

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Input tensor should have 3 dimensions [max_time, batch_size, num_units].")
        max_time, batch_size, num_units = input_shape
        if num_units is None:
            raise ValueError("Last dimension of input tensor should be known")
            
        output_max_time = None
        if max_time is not None:
            output_max_time = max_time // factor + 1 if max_time % factor > 0 else 0
        
        output_shape = (output_max_time, batch_size, num_units)
        return output_shape
    
    def get_config(self):
        return {"factor": self._factor, "time_major": self._time_major}


class EncoderNetwork(keras.layers.Layer):
    def __init__(self, 
                 num_layers,
                 lstm_size,
                 projection_size,
                 reduction_indexes=[],
                 convert_tflite=False):
        super().__init__()
        self.num_layers = num_layers
        
        # Define sublayers
        self.lstms = []
        self.layer_norms = []
        self.projections = []
        self.time_reductions = {}
        for i in range(num_layers):
            self.lstms.append(
                layers.LSTM(lstm_size,
                        return_sequences=True,
                        return_state=True,
                        time_major=True,
                        unroll=convert_tflite))
            self.projections.append(
                layers.Dense(projection_size))
            self.layer_norms.append(
                layers.LayerNormalization())
            if i in reduction_indexes:
                # Warning originally there was a wrong time reduction incompatible with time_major tensors
                self.time_reductions[str(i)] = TimeReductionLayer(factor=2, time_major=True)
                
    def call(self, x, training=False, states=None):
        if states is None:
            states = [None] * len(self.lstms)
        new_states = []
        
        x = tf.transpose(x, [1, 0, 2])
        for i in range(self.num_layers):
            (x, c_state, h_state) = self.lstms[i](x, initial_state=states[i], training=training)
            new_states.append([c_state, h_state])
            x = self.projections[i](x, training=training)
            x = self.layer_norms[i](x, training=training)
            
            if str(i) in self.time_reductions:
                x = self.time_reductions[str(i)](x)
        x = tf.transpose(x, [1, 0, 2])
        
        return x, new_states

class PredictionNetwork(keras.layers.Layer):
    def __init__(self, 
                 num_layers,
                 lstm_size,
                 projection_size,
                 convert_tflite=False):
        super().__init__()
        self.num_layers = num_layers
        
        # Define sublayers
        self.lstms = []
        self.layer_norms = []
        self.projections = []
        for i in range(num_layers):
            self.lstms.append(
                layers.LSTM(lstm_size,
                        return_sequences=True,
                        return_state=True,
                        time_major=True,
                        unroll=convert_tflite))
            self.projections.append(
                layers.Dense(projection_size))
            self.layer_norms.append(
                layers.LayerNormalization())
                
    def call(self, x, training=False, states=None):
        if states is None:
            states = [None] * len(self.lstms)
        new_states = []
        
        x = tf.transpose(x, [1, 0, 2])
        for i in range(self.num_layers):
            (x, c_state, h_state) = self.lstms[i](x, initial_state=states[i], training=training)
            new_states.append([c_state, h_state])
            x = self.projections[i](x, training=training)
            x = self.layer_norms[i](x, training=training)
        x = tf.transpose(x, [1, 0, 2])
        
        return x, new_states
    
class JointNetwork(keras.layers.Layer):
    def __init__(self, 
                 output_size,
                 additional_size=None,
                 aggregation_type='sum'):
        super().__init__()        
        # Define sublayers
        self.linear_0 = None
        if additional_size is not None:
            self.linear_0 = keras.layers.Dense(additional_size)
        self.linear = keras.layers.Dense(output_size)
        
        assert aggregation_type in ['sum', 'concat']
        self.aggregation_type = aggregation_type    
            
    def call(self, x, training=False):
        pred_inp = x[0]
        enc_inp = x[1]
        
        # [B, T, V] => [B, T, U, V]
        # [B, U, V] => [B, T, U, V]
        if self.aggregation_type == 'sum':
            x = (tf.tile(tf.expand_dims(enc_inp, 2), [1, 1, tf.shape(pred_inp)[1], 1]) +
                 tf.tile(tf.expand_dims(pred_inp, 1), [1, tf.shape(enc_inp)[1], 1, 1]))
        else:
            x = tf.concat([tf.tile(tf.expand_dims(enc_inp, 2), [1, 1, tf.shape(pred_inp)[1], 1]),
                           tf.tile(tf.expand_dims(pred_inp, 1), [1, tf.shape(enc_inp)[1], 1, 1])],
                          axis=-1)
        
        if self.linear_0 is not None:
            x = self.linear_0(x)
        x = self.linear(x)
        return x
    

def get_rnnt(encoder_input_size,
             num_layers_encoder=2,
             units_encoder=20,
             projection_encoder=10,
             encoder_reduction_indexes=[0],
             units_prediction=20,
             projection_prediction=10,
             num_layers_prediction=2,
             joint_additional_size=None,
             joint_aggregation_type='sum',
             vocab_size=10,
             blank_label=0,
             convert_tflite=False, random_state=10) -> keras.Model:
    max_seq_length = None
    if convert_tflite:
        max_seq_length = 50

    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    # Create model under CPU scope and avoid OOM, errors during concatenation
    # a large distributed model.
    # Define input tensor [batch, time, features]
    input_tensor = layers.Input([max_seq_length, encoder_input_size], name='features')
    labels = tf.keras.Input(shape=[max_seq_length], dtype=tf.int32, name='labels')

    encoder_result, _ = EncoderNetwork(
        num_layers=num_layers_encoder,
        lstm_size=units_encoder,
        projection_size=projection_encoder,
        reduction_indexes=encoder_reduction_indexes,
        convert_tflite=convert_tflite)(input_tensor)
    
    label_input = tf.one_hot(labels, vocab_size)
    # Prepend zero vector as additional timestep
    zero_timestep_padding = [[0, 0], [1, 0], [0, 0]]
    label_input = tf.pad(label_input, zero_timestep_padding)
    
    prediction_result, _ = PredictionNetwork(
                 num_layers=num_layers_prediction,
                 projection_size=projection_prediction,
                 lstm_size=units_prediction)(label_input)
    
    # TODO resolve prepending <sos> symbol
    # Loss requires first step to be zero vector
    outputs = JointNetwork(vocab_size, 
                           additional_size=joint_additional_size,
                           aggregation_type=joint_aggregation_type)([prediction_result, encoder_result])

    if convert_tflite:
        return keras.Model([input_tensor, labels], outputs, name='RNNT')
    else:
        # Having 1 element vector is required to save and load model in non nightly tensorflow
        # https://github.com/tensorflow/tensorflow/issues/35446.
        feature_lengths = tf.keras.Input(shape=[1], dtype=tf.int32, name='feature_lengths')
        label_lengths = tf.keras.Input(shape=[1], dtype=tf.int32, name='label_lengths')        
        
        model = keras.Model([input_tensor, labels, feature_lengths, label_lengths], outputs, name='RNNT')
        model.add_loss(tf.reduce_mean(rnnt_loss(outputs,
                                                labels,
                                                (tf.math.floordiv(feature_lengths, 2) +  tf.math.floormod(feature_lengths, 2))[:, 0],
                                                label_lengths[:, 0],
                                                blank_label=blank_label)
        ))
        
        return model
