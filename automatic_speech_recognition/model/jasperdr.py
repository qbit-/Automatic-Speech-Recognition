"""
This module implements Jasper model as described in https://arxiv.org/pdf/1904.03288.pdf
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision


class Small_block(keras.Model):
    """
    Conv 1D convolutional module
    """
    def __init__(self, kernel_size, filters, layer_name,
                 residual=False, use_biases=False, use_batchnorms=True):
        super(Small_block, self).__init__(name=layer_name)
        self.conv = layers.Conv1D(filters, kernel_size, padding='same',
                                  use_bias=use_biases, bias_initializer='zeros', name='conv')
        if use_batchnorms: self.bn = layers.BatchNormalization(momentum=0.9)
        else: self.bn = None
        self.residual = residual
        self.relu = layers.ReLU()
        self.kernel_size = kernel_size
        self.filters = filters
        self.layer_name = layer_name
        self.use_biases = use_biases
        self.use_batchnorms = use_batchnorms

    def call(self, input_tensor, res_values, training=False):
        x = self.conv(input_tensor)
        if self.bn is not None: 
            x = self.bn(x, training=training)
        if self.residual:
            x = tf.keras.layers.add([x] + res_values)
        x = self.relu(x)
        tf.debugging.assert_all_finite(x, f'nan after {self.layer_name}', name=None)
        return x

    def get_config(self):
        config = {}
        config.update(
            {
                'kernel_size': self.kernel_size,
                'filters': self.filters,
                'residual': self.residual,
                'layer_name': self.layer_name,
                'use_biases': self.use_biases,
                'use_batchnorms': self.use_batchnorms
            }
        )
        return config


class B_block(keras.Model):
    """
    Base residual block of the Quartznet model
    """
    def __init__(self, kernel_size, filters, n_small_blocks, num_res_connections, layer_name,
                 use_biases=False, use_batchnorms=True):
        super(B_block, self).__init__(name=layer_name)
        self.small_blocks = []
        for i in range(n_small_blocks - 1):
            self.small_blocks.append(Small_block(kernel_size, filters, layer_name='SB-{}'.format(i),
                                                 use_biases=use_biases, use_batchnorms=use_batchnorms))
        self.res_block = Small_block(kernel_size, filters, layer_name='SB-res', residual=True,
                                     use_biases=use_biases, use_batchnorms=use_batchnorms)
        self.res_convs = [layers.Conv1D(
            filters, 1, padding='same', use_bias=use_biases, bias_initializer='zeros', name=f'res_conv_{i}')
            for i in range(num_res_connections)]
        if use_batchnorms: 
            self.bns = [layers.BatchNormalization(
                momentum=0.9, name=f'res_bn_{i}') for i in range(num_res_connections)]
        else: 
            self.bns = None
        self.kernel_size = kernel_size
        self.filters = filters
        self.n_small_blocks = n_small_blocks
        self.num_res_connections = num_res_connections
        self.layer_name = layer_name
        self.use_biases = use_biases
        self.use_batchnorms = use_batchnorms

    def call(self, x, res_inputs, training=False):
        res_values = [self.res_convs[i](y) for i, y in enumerate(res_inputs)]
        if self.bns is not None: 
            res_values = [self.bns[i](y, training=training) for i, y in enumerate(res_values)]
        for i in range(len(self.small_blocks)):
            x = self.small_blocks[i](x, None, training=training)
        x = self.res_block(x, res_values, training=training)
        tf.debugging.assert_all_finite(x, f'nan after {self.layer_name}', name=None)
        return x

    def get_config(self):
        config = {}
        config.update(
            {
                'kernel_size': self.kernel_size,
                'filters': self.filters,
                'n_small_blocks': self.n_small_blocks,
                'num_res_connections': self.num_res_connections,
                'layer_name': self.layer_name,
                'use_biases': self.use_biases,
                'use_batchnorms': self.use_batchnorms
            }
        )
        return config
    
    @classmethod
    def from_config(cls, config):
        # in tensorflow2.2 instead of keras.layer.from_config method the engine.network.from_config is used
        return cls(**config)



def get_jasperdr(input_dim, output_dim,
              is_mixed_precision=False,
              tflite_version=False,
              num_b_block_repeats=3,
              b_block_kernel_sizes=(11, 13, 17, 21, 25),
              b_block_num_channels=(256, 384, 512, 640, 768),
              num_small_blocks=5,
              use_biases=False,
              use_batchnorms=True,
              use_mask=False,
              fixed_batch_size=None,
              random_state=1) -> keras.Model:
    """
    Parameters
    ----------
    input_dim: input feature length
    output_dim: output feature length
    is_mixed_precision: if mixed precision model is needed
    tflite_version: if export to tflite is needed
    num_b_block_repeats: 1 is 5x5 quartznet, 2 is 10x5, 3 is 15x5
    b_block_kernel_sizes: iterable, kernel size of each b block
    b_block_num_channels: iterable, number of channels of each b block
    """
    assert len(b_block_kernel_sizes) == len(b_block_num_channels), \
        "Number of kernel sizes not equal the number of channel sizes"

    max_seq_length = None
    if tflite_version:
        max_seq_length = 10

    if is_mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    with tf.device('/cpu:0'):
        if fixed_batch_size is None:
            input_tensor = layers.Input(shape=[max_seq_length, input_dim], 
                                        name='X')
        else:
            input_tensor = layers.Input(shape=[max_seq_length, input_dim], 
                                        name='X', 
                                        batch_size=fixed_batch_size)
        x = tf.identity(input_tensor)
        if use_mask: x = layers.Masking()(x)
        # First encoder layer
        x = layers.Conv1D(
            256, 11, padding='same', strides=2,
            name='conv_1', use_bias=use_biases, bias_initializer='zeros')(x)
        if use_batchnorms: x = layers.BatchNormalization(name='BN-1', momentum=0.9)(x)
        x = layers.ReLU(name='RELU-1')(x)

        block_idx = 1
        res_inputs = []
        for kernel_size, n_channels in zip(
                b_block_kernel_sizes, b_block_num_channels):
            for bk in range(num_b_block_repeats):
                res_inputs = res_inputs + [x]
                x = B_block(
                    kernel_size, n_channels, 
                    num_small_blocks, 
                    len(res_inputs),
                    f'B-{block_idx}',
                    use_biases=use_biases, 
                    use_batchnorms=use_batchnorms)(x, res_inputs)
                block_idx += 1
                
        # First final layer
        x = layers.Conv1D(
            896, 29, padding='same', name='conv_2',
            dilation_rate=2, use_bias=use_biases, bias_initializer='zeros')(x)
        if use_batchnorms: x = layers.BatchNormalization(name='BN-2', momentum=0.9)(x)
        x = layers.ReLU(name='RELU-2')(x)

        # Second final layer
        x = layers.Conv1D(1024, 1, padding='same',
                          name='conv_3', use_bias=use_biases, bias_initializer='zeros')(x)
        if use_batchnorms: x = layers.BatchNormalization(name='BN-3', momentum=0.9)(x)
        x = layers.ReLU(name='RELU-3')(x)

        # Third final layer
        x = layers.Conv1D(
            output_dim, 1, padding='same', dilation_rate=1, name='conv_4')(x)
        model = keras.Model([input_tensor], x, name='Jasper_dr')

    if is_mixed_precision:
        policy = mixed_precision.Policy('float32')
        mixed_precision.set_policy(policy)

    return model


QUARTZNET_LAYERS = {'Small_block': Small_block, 'B_block': B_block}


def load_nvidia_jasperdr(
        enc_path="./data/JasperEncoder_3-STEP-218410.pt",
        dec_path="./data/JasperDecoderForCTC_4-STEP-218410.pt",
        use_biases=False,
        tflite_version=False,
        fixed_batch_size=None):
    """
    pass paths to these files as decoder and encoder paths
    """
    import torch
    model = get_jasperdr(input_dim=64, output_dim=29,
                          is_mixed_precision=False,
                          tflite_version=tflite_version,
                          num_b_block_repeats=2,
                          b_block_kernel_sizes=(11, 13, 17, 21, 25),
                          b_block_num_channels=(256, 384, 512, 640, 768),
                          num_small_blocks=5,
                          use_biases=False,
                          use_batchnorms=True,
                          fixed_batch_size=fixed_batch_size,
                          random_state=1)

    enc = torch.load(enc_path, map_location=torch.device('cpu'))
    dec = torch.load(dec_path, map_location=torch.device('cpu'))

    # First encoder layer
    conv_1 = model.get_layer(name='conv_1')
    conv_1.set_weights([
        enc['encoder.0.conv.0.weight'].cpu().permute(2, 1, 0).numpy()])
    BN_1 = model.get_layer(name='BN-1')
    BN_1.set_weights([
        enc['encoder.0.conv.1.weight'].cpu().numpy(),
        enc['encoder.0.conv.1.bias'].cpu().numpy(),
        enc['encoder.0.conv.1.running_mean'].cpu().numpy(),
        enc['encoder.0.conv.1.running_var'].cpu().numpy()
    ])

    for i in range(1, 11):
        layer_name = f'B-{i}'
        b_block = model.get_layer(name=layer_name)
        new_weights = [
            enc[(f'encoder.{i}.conv.0.weight')].cpu().permute(
                2, 1, 0).numpy(),
            enc[(f'encoder.{i}.conv.1.weight')].cpu().numpy(),
            enc[(f'encoder.{i}.conv.1.bias')].cpu().numpy(),
            enc[(f'encoder.{i}.conv.4.weight')].cpu().permute(
                2, 1, 0).numpy(),
            enc[(f'encoder.{i}.conv.5.weight')].cpu().numpy(),
            enc[(f'encoder.{i}.conv.5.bias')].cpu().numpy(),
            enc[(f'encoder.{i}.conv.8.weight')].cpu().permute(
                2, 1, 0).numpy(),
            enc[(f'encoder.{i}.conv.9.weight')].cpu().numpy(),
            enc[(f'encoder.{i}.conv.9.bias')].cpu().numpy(),
            enc[(f'encoder.{i}.conv.12.weight')].cpu().permute(
                2, 1, 0).numpy(),
            enc[(f'encoder.{i}.conv.13.weight')].cpu().numpy(),
            enc[(f'encoder.{i}.conv.13.bias')].cpu().numpy(),

            enc[(f'encoder.{i}.conv.1.running_mean')].cpu().numpy(),
            enc[(f'encoder.{i}.conv.1.running_var')].cpu().numpy(),

            enc[(f'encoder.{i}.conv.5.running_mean')].cpu().numpy(),
            enc[(f'encoder.{i}.conv.5.running_var')].cpu().numpy(),

            enc[(f'encoder.{i}.conv.9.running_mean')].cpu().numpy(),
            enc[(f'encoder.{i}.conv.9.running_var')].cpu().numpy(),

            enc[(f'encoder.{i}.conv.13.running_mean')].cpu().numpy(),
            enc[(f'encoder.{i}.conv.13.running_var')].cpu().numpy(),
            enc[(f'encoder.{i}.conv.16.weight')].cpu().permute(2, 1, 0).numpy(),
            enc[(f'encoder.{i}.conv.17.weight')].cpu().numpy(),
            enc[(f'encoder.{i}.conv.17.bias')].cpu().numpy(),
            enc[(f'encoder.{i}.conv.17.running_mean')].cpu().numpy(),
            enc[(f'encoder.{i}.conv.17.running_var')].cpu().numpy()]
            
        # 1x1 convs of inputs from previous blocks
        for j in range(i):
            new_weights.extend([
                enc[(f'encoder.{i}.res.{j}.0.weight')].cpu().permute(
                    2, 1, 0).numpy()
            ])
            
        # batchnorms of 1x1 convs of inputs from previous blocks
        for j in range(i):
            new_weights.extend([
                enc[(f'encoder.{i}.res.{j}.1.weight')].cpu().numpy(),
                enc[(f'encoder.{i}.res.{j}.1.bias')].cpu().numpy(),
            ])
        for j in range(i):
            new_weights.extend([
                enc[(f'encoder.{i}.res.{j}.1.running_mean')].cpu().numpy(),
                enc[(f'encoder.{i}.res.{j}.1.running_var')].cpu().numpy()
            ])
        
        b_block.set_weights(new_weights)

    # First final layer
    conv_2 = model.get_layer(name='conv_2')
    conv_2.set_weights([
        enc['encoder.11.conv.0.weight'].cpu().permute(
            2, 1, 0).numpy()])
    BN_2 = model.get_layer(name='BN-2')
    BN_2.set_weights([
        enc['encoder.11.conv.1.weight'].cpu().numpy(),
        enc['encoder.11.conv.1.bias'].cpu().numpy(),
        enc['encoder.11.conv.1.running_mean'].cpu().numpy(),
        enc['encoder.11.conv.1.running_var'].cpu().numpy()
    ])

    # Second final layer
    conv_3 = model.get_layer(name='conv_3')
    conv_3.set_weights([
        enc['encoder.12.conv.0.weight'].cpu().permute(2, 1, 0).numpy()])
    BN_3 = model.get_layer(name='BN-3')
    BN_3.set_weights([
        enc['encoder.12.conv.1.weight'].cpu().numpy(),
        enc['encoder.12.conv.1.bias'].cpu().numpy(),
        enc['encoder.12.conv.1.running_mean'].cpu().numpy(),
        enc['encoder.12.conv.1.running_var'].cpu().numpy()
    ])

    # Third final layer
    conv_4 = model.get_layer(name='conv_4')
    conv_4.set_weights(
        [dec['decoder_layers.0.weight'].cpu().permute(
            2, 1, 0).numpy(),
         dec['decoder_layers.0.bias'].cpu().numpy()])
    
    return model
