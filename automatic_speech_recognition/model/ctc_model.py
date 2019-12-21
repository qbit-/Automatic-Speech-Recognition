from typing import List, Tupleimport numpy as npimport tensorflow as tffrom tensorflow import kerasfrom tensorflow.keras import layersdef get_ctc_model(layers_params: List[dict],                  input_dim: int,                  to_freeze: Tuple[dict] = (),                  random_state: int = 1) -> keras.Model:    np.random.seed(random_state)    tf.random.set_seed(random_state)    constructors = {        'BatchNormalization':            lambda params: layers.BatchNormalization(**params),        'Conv2D':            lambda params: layers.Conv2D(**params, name=name),        'Dense':            lambda params: layers.TimeDistributed(layers.Dense(**params),                                                  name=name),        'Dropout':            lambda params: layers.Dropout(**params),        'LSTM':            lambda params: layers.Bidirectional(layers.LSTM(**params),                                                merge_mode='sum', name=name),        'ReLU':            lambda params: layers.ReLU(**params),        'ZeroPadding2D':            lambda params: layers.ZeroPadding2D(**params),        'expand_dims':            lambda params: layers.Lambda(keras.backend.expand_dims,                                         arguments=params),        'squeeze':            lambda params: layers.Lambda(keras.backend.squeeze,                                         arguments=params),        'squeeze_last_dims':            lambda params: layers.Reshape([-1, params['units']])    }    # Create model under CPU scope and avoid OOM, errors during concatenation    # a large distributed model.    with tf.device('/cpu:0'):        # Define input tensor [time, features]        input_tensor = layers.Input([None, input_dim], name='X')        x = input_tensor        for params in layers_params:            constructor_name = params.pop('constructor')            # `name` is implicit passed to constructors Conv2D,            # TimeDistributed and Bidirectional.            name = params.pop('name') if 'name' in params else None            constructor = constructors[constructor_name]            layer = constructor(params)            x = layer(x)        # Return at each time step logits along characters. Then CTC        # computation is more stable, in contrast to the softmax.        output_tensor = x        model = keras.Model(input_tensor, output_tensor, name='DeepSpeech')        for params in to_freeze:            name = params.pop('name')            layer = model.get_layer(name)            layer.trainable = False    return model