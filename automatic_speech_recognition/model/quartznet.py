"""
This module implements Quartznet model as describe in http://arxiv.org/abs/1910.10261
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision


class Small_block(keras.Model):
    """
    Time-channel separable 1D convolutional module
    """
    def __init__(self, kernel_size, filters, layer_name,
                 residual=False, use_biases=False, use_batchnorms=True):
        super(Small_block, self).__init__(name=layer_name)
        self.conv = layers.SeparableConv1D(
            filters, kernel_size, padding='same', 
            use_bias=use_biases, bias_initializer='zeros', name='sepconv')
        if use_batchnorms: self.bn = layers.BatchNormalization(momentum=0.9)
        else: self.bn = None
        self.residual = residual
        self.relu = layers.ReLU()
        self.kernel_size = kernel_size
        self.filters = filters
        self.layer_name = layer_name
        self.use_biases = use_biases
        self.use_batchnorms = use_batchnorms

    def call(self, input_tensor, residual_value, training=False):
        x = self.conv(input_tensor)
        if self.bn is not None: x = self.bn(x, training=training)
        if self.residual:
            x += residual_value
        x = self.relu(x)
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
    def __init__(self, kernel_size, filters, n_small_blocks, layer_name, 
                 use_biases=False, use_batchnorms=True):
        super(B_block, self).__init__(name=layer_name)
        self.small_blocks = []
        for i in range(n_small_blocks - 1):
            self.small_blocks.append(Small_block(kernel_size, filters, layer_name='SB-{}'.format(i),
                                                 use_biases=use_biases, use_batchnorms=use_batchnorms))
        self.res_block = Small_block(kernel_size, filters, layer_name='SB-res', residual=True,
                                     use_biases=use_biases, use_batchnorms=use_batchnorms)
        self.conv = layers.Conv1D(
            filters, 1, padding='same', use_bias=use_biases, bias_initializer='zeros', name='conv')
        if use_batchnorms: self.bn = layers.BatchNormalization(momentum=0.9)
        else: self.bn = None
        self.kernel_size = kernel_size
        self.filters = filters
        self.n_small_blocks = n_small_blocks
        self.layer_name = layer_name
        self.use_biases = use_biases
        self.use_batchnorms = use_batchnorms

    def call(self, x, training=False):
        residual_value = self.conv(x)
        if self.bn is not None: residual_value = self.bn(residual_value, training=training)
        for i in range(len(self.small_blocks)):
            x = self.small_blocks[i](x, None, training=training)
        x = self.res_block(x, residual_value, training=training)
        return x

    def get_config(self):
        config = {}
        config.update(
            {
                'kernel_size': self.kernel_size,
                'filters': self.filters,
                'n_small_blocks': self.n_small_blocks,
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



def get_quartznet(input_dim, output_dim,
                  is_mixed_precision=False,
                  fixed_sequence_size=None,
                  num_b_block_repeats=3,
                  b_block_kernel_sizes=(33, 39, 51, 63, 75),
                  b_block_num_channels=(256, 256, 512, 512, 512),
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
    fixed_sequence_size: int, default None. If the length of sequence has to be fixed.
    num_b_block_repeats: 1 is 5x5 quartznet, 2 is 10x5, 3 is 15x5
    b_block_kernel_sizes: iterable, kernel size of each b block
    b_block_num_channels: iterable, number of channels of each b block
    num_small_blocks: int, number of conv blocks inside 1 residual block
    use_biases: if biases are used with convolutions
    use_batchnorms: if batchnorms are inserted after each residual block
    use_mask: if mask layer is used
    fixed_batch_size: int, default None. If the model will have fixed batch size
    random_state: int, state used for weight initialization
     """
    assert len(b_block_kernel_sizes) == len(b_block_num_channels), \
        "Number of kernel sizes not equal the number of channel sizes"

    max_seq_length = None
    if tflite_version:
        max_seq_length = 5

    if is_mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    with tf.device('/cpu:0'):
        input_tensor = layers.Input([fixed_sequence_size, input_dim], name='X',
                                    batch_size=fixed_batch_size)
        x = tf.identity(input_tensor)
        if use_mask: x = layers.Masking()(x)
        # First encoder layer
        x = layers.SeparableConv1D(
            256, 33, padding='same', strides=2,
            name='conv_1', use_bias=use_biases, bias_initializer='zeros')(x)
        if use_batchnorms: x = layers.BatchNormalization(name='BN-1', momentum=0.9)(x)
        x = layers.ReLU(name='RELU-1')(x)

        block_idx = 1
        for kernel_size, n_channels in zip(
                b_block_kernel_sizes, b_block_num_channels):
            for bk in range(num_b_block_repeats):
                x = B_block(
                    kernel_size, n_channels, num_small_blocks,
                    f'B-{block_idx}',
                    use_biases=use_biases, 
                    use_batchnorms=use_batchnorms)(x)
                block_idx += 1

        # First final layer
        x = layers.SeparableConv1D(
            512, 87, padding='same', name='conv_2',
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
        model = keras.Model([input_tensor], x, name='QuartzNet')

    if is_mixed_precision:
        policy = mixed_precision.Policy('float32')
        mixed_precision.set_policy(policy)

    return model


QUARTZNET_LAYERS = {'Small_block': Small_block, 'B_block': B_block}


def load_nvidia_quartznet(
        enc_path="./data/JasperEncoder-STEP-247400.pt",
        dec_path="./data/JasperDecoderForCTC-STEP-247400.pt",
        use_biases=False):
    """
    The weights for Quartznet model (English)
    can be downloaded with the following command:
    curl -LO https://api.ngc.nvidia.com/v2/models/nvidia/quartznet15x5/versions/2/files/quartznet15x5/JasperDecoderForCTC-STEP-247400.pt
    curl -LO https://api.ngc.nvidia.com/v2/models/nvidia/quartznet15x5/versions/2/files/quartznet15x5/JasperEncoder-STEP-247400.pt

    pass paths to these files as decoder and encoder paths
    """
    import torch
    model = get_quartznet(input_dim=64, output_dim=29,
                          is_mixed_precision=False,
                          tflite_version=False,
                          num_b_block_repeats=3,
                          b_block_kernel_sizes=(33, 39, 51, 63, 75),
                          b_block_num_channels=(256, 256, 512, 512, 512),
                          num_small_blocks=5,
                          use_biases=use_biases,
                          use_batchnorms=True,
                          random_state=1)

    enc = torch.load(enc_path, map_location=torch.device('cpu'))
    dec = torch.load(dec_path, map_location=torch.device('cpu'))

    # First encoder layer
    conv_1 = model.get_layer(name='conv_1')
    new_weights = [
        enc['encoder.0.mconv.0.conv.weight'].cpu().permute(
            2, 0, 1).numpy(),
        enc['encoder.0.mconv.1.conv.weight'].cpu().permute(
             2, 1, 0).numpy()]
    if use_biases: new_weights.append(conv_1.bias.numpy())
    conv_1.set_weights(new_weights)
    BN_1 = model.get_layer(name='BN-1')
    BN_1.set_weights([
        enc['encoder.0.mconv.2.weight'].cpu().numpy(),
        enc['encoder.0.mconv.2.bias'].cpu().numpy(),
        enc['encoder.0.mconv.2.running_mean'].cpu().numpy(),
        enc['encoder.0.mconv.2.running_var'].cpu().numpy()
    ])

    for i in range(1, 16):
        layer_name = f'B-{i}'
        b_block = model.get_layer(name=layer_name)
        new_weights = [
            enc[(f'encoder.{i}.mconv.0.conv.weight')].cpu().permute(
                2, 0, 1).numpy(),
            enc[(f'encoder.{i}.mconv.1.conv.weight')].cpu().permute(
                2, 1, 0).numpy()]
        if use_biases: new_weights.append(tf.zeros(b_block.layers[0].layers[0].bias.shape))
        new_weights.extend([
            enc[(f'encoder.{i}.mconv.2.weight')].cpu().numpy(),
            enc[(f'encoder.{i}.mconv.2.bias')].cpu().numpy(),
            enc[(f'encoder.{i}.mconv.5.conv.weight')].cpu().permute(
                2, 0, 1).numpy(),
            enc[(f'encoder.{i}.mconv.6.conv.weight')].cpu().permute(
                2, 1, 0).numpy()])
        if use_biases: new_weights.append(tf.zeros(b_block.layers[0].layers[0].bias.shape))
        new_weights.extend([
            enc[(f'encoder.{i}.mconv.7.weight')].cpu().numpy(),
            enc[(f'encoder.{i}.mconv.7.bias')].cpu().numpy(),
            enc[(f'encoder.{i}.mconv.10.conv.weight')].cpu().permute(
                2, 0, 1).numpy(),
            enc[(f'encoder.{i}.mconv.11.conv.weight')].cpu().permute(
                2, 1, 0).numpy()])
        if use_biases: new_weights.append(tf.zeros(b_block.layers[0].layers[0].bias.shape))
        new_weights.extend([
            enc[(f'encoder.{i}.mconv.12.weight')].cpu().numpy(),
            enc[(f'encoder.{i}.mconv.12.bias')].cpu().numpy(),
            enc[(f'encoder.{i}.mconv.15.conv.weight')].cpu().permute(
                2, 0, 1).numpy(),
            enc[(f'encoder.{i}.mconv.16.conv.weight')].cpu().permute(
                2, 1, 0).numpy()])
        if use_biases: new_weights.append(tf.zeros(b_block.layers[0].layers[0].bias.shape))
        new_weights.extend([
            enc[(f'encoder.{i}.mconv.17.weight')].cpu().numpy(),
            enc[(f'encoder.{i}.mconv.17.bias')].cpu().numpy(),

            enc[(f'encoder.{i}.mconv.2.running_mean')].cpu().numpy(),
            enc[(f'encoder.{i}.mconv.2.running_var')].cpu().numpy(),

            enc[(f'encoder.{i}.mconv.7.running_mean')].cpu().numpy(),
            enc[(f'encoder.{i}.mconv.7.running_var')].cpu().numpy(),

            enc[(f'encoder.{i}.mconv.12.running_mean')].cpu().numpy(),
            enc[(f'encoder.{i}.mconv.12.running_var')].cpu().numpy(),

            enc[(f'encoder.{i}.mconv.17.running_mean')].cpu().numpy(),
            enc[(f'encoder.{i}.mconv.17.running_var')].cpu().numpy(),

            enc[(f'encoder.{i}.mconv.20.conv.weight')].cpu().permute(
                2, 0, 1).numpy(),
            enc[(f'encoder.{i}.mconv.21.conv.weight')].cpu().permute(
                2, 1, 0).numpy()])
        if use_biases: new_weights.append(tf.zeros(b_block.layers[0].layers[0].bias.shape))
        new_weights.extend([
            enc[(f'encoder.{i}.mconv.22.weight')].cpu().numpy(),
            enc[(f'encoder.{i}.mconv.22.bias')].cpu().numpy(),
            enc[(f'encoder.{i}.mconv.22.running_mean')].cpu().numpy(),
            enc[(f'encoder.{i}.mconv.22.running_var')].cpu().numpy(),
            # residual small block
            enc[(f'encoder.{i}.res.0.0.conv.weight')].cpu().permute(
                2, 1, 0).numpy()])
        if use_biases: new_weights.append(b_block.layers[-2].bias.numpy())
        new_weights.extend([
            enc[(f'encoder.{i}.res.0.1.weight')].cpu().numpy(),
            enc[(f'encoder.{i}.res.0.1.bias')].cpu().numpy(),
            enc[(f'encoder.{i}.res.0.1.running_mean')].cpu().numpy(),
            enc[(f'encoder.{i}.res.0.1.running_var')].cpu().numpy()])
        b_block.set_weights(new_weights)

    # First final layer
    conv_2 = model.get_layer(name='conv_2')
    new_weights = [
        enc['encoder.16.mconv.0.conv.weight'].cpu().permute(
            2, 0, 1).numpy(),
        enc['encoder.16.mconv.1.conv.weight'].cpu().permute(
             2, 1, 0).numpy()]
    if use_biases: new_weights.append(conv_2.bias.numpy())
    conv_2.set_weights(new_weights)

    BN_2 = model.get_layer(name='BN-2')
    BN_2.set_weights([
        enc['encoder.16.mconv.2.weight'].cpu().numpy(),
        enc['encoder.16.mconv.2.bias'].cpu().numpy(),
        enc['encoder.16.mconv.2.running_mean'].cpu().numpy(),
        enc['encoder.16.mconv.2.running_var'].cpu().numpy()
    ])

    # Second final layer
    conv_3 = model.get_layer(name='conv_3')
    new_weights = [
        enc['encoder.17.mconv.0.conv.weight'].cpu().permute(
            2, 1, 0).numpy()]
    if use_biases: new_weights.append(conv_3.bias.numpy())
    conv_3.set_weights(new_weights)
    BN_3 = model.get_layer(name='BN-3')
    BN_3.set_weights([
        enc['encoder.17.mconv.1.weight'].cpu().numpy(),
        enc['encoder.17.mconv.1.bias'].cpu().numpy(),
        enc['encoder.17.mconv.1.running_mean'].cpu().numpy(),
        enc['encoder.17.mconv.1.running_var'].cpu().numpy()
    ])

    # Third final layer
    conv_4 = model.get_layer(name='conv_4')
    conv_4.set_weights(
        [dec['decoder_layers.0.weight'].cpu().permute(
            2, 1, 0).numpy(),
         dec['decoder_layers.0.bias'].cpu().numpy()])
    return model
