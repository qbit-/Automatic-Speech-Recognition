
import tensorflow as tf
import tensorflow_addons as tfa
import sys
import os
if os.path.abspath('../') not in sys.path:
    sys.path.append(os.path.abspath('../'))

import automatic_speech_recognition as asr
import time
from datetime import datetime
import argparse
import pickle


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.mixed_precision import experimental as mixed_precision


# In[4]:


from automatic_speech_recognition.model.quartznet import QUARTZNET_LAYERS


# In[5]:


import horovod.tensorflow.keras as hvd



# ## Train

# In[8]:


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# In[9]:


# Initialize Horovod
hvd.init()
# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    local_rank = hvd.local_rank()
    tf.config.experimental.set_visible_devices(gpus[local_rank], 'GPU')
    print(f'rank {local_rank} will use GPU {gpus[local_rank]}')

# In[10]:


def get_pipeline(model, optimizer=None):
    alphabet = asr.text.Alphabet(lang='en')
    features_extractor = asr.features.FilterBanks(
        features_num=64,
        standardize=None,
        winlen=0.02,
        winstep=0.01,
    )
    if not optimizer:
        optimizer = tf.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
    decoder = asr.decoder.GreedyDecoder()
    pipeline = asr.pipeline.CTCPipeline(
        alphabet, features_extractor, model, optimizer, decoder
    )
    return pipeline


# In[ ]:





# In[12]:


def train_model(filename, dataset_idx, val_dataset_idx=None, initial_lr=0.001, batch_size=10, epochs=25, tensorboard=False, restart_filename=None,
               is_mixed_precision=False, n_blocks=1):
    basename = os.path.basename(filename).split('.')[0]
    model_dir = os.path.join(os.path.dirname(filename), basename + '_train')
    os.makedirs(model_dir, exist_ok=True)
    
    model = asr.model.get_quartznet(64, 29, is_mixed_precision=is_mixed_precision, num_b_block_repeats=n_blocks)

    if restart_filename:
        model.load_weights(restart_filename)

    batch_size_global = batch_size * hvd.size()
    initial_lr_global = initial_lr * hvd.size()
    print(f'batch_size: {batch_size_global}')
    print(f'initial_lr: {initial_lr_global}')

    dataset = asr.dataset.Audio.from_csv(dataset_idx, batch_size=batch_size_global, use_filesizes=True)
    dataset.sort_by_length()
    dataset.shuffle_indices()
    if val_dataset_idx:
        val_dataset = asr.dataset.Audio.from_csv(val_dataset_idx, batch_size=batch_size_global, use_filesizes=True)

    opt_instance = tf.optimizers.Adam(initial_lr_global, beta_1=0.9, beta_2=0.999)
    #opt_instance = tfa.optimizers.NovoGrad(initial_lr_global, beta_1=0.95, beta_2=0.5, weight_decay=0.001)

    opt = hvd.DistributedOptimizer(opt_instance)
    pipeline = get_pipeline(model, opt)
    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
    ]
    schedule=tf.keras.experimental.CosineDecayRestarts(
        initial_lr_global, 10, t_mul=2.0, m_mul=1.0, alpha=0.0,
    )
    callbacks.append(LearningRateScheduler(schedule))
    if hvd.rank() == 0:
        prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
        monitor_metric_name = 'loss' if not val_dataset_idx else 'val_loss'  # val_loss is wrong and broken
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                os.path.join(model_dir, prefix + '_best.h5'),
                monitor=monitor_metric_name, save_weights_only=True,
                save_best_only=True))
        if tensorboard:
            logdir = os.path.join(model_dir, 'tb', prefix)
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
            callbacks.append(tensorboard_callback)

    time_start = time.time()

    hist = pipeline.fit(dataset, epochs=epochs, dev_dataset=val_dataset,
                        #steps_per_epoch=270,
                        callbacks=callbacks,
                        verbose=1 if hvd.rank() == 0 else 0,
                        #workers=2, use_multiprocessing=True
    )
    elapsed = time.time() - time_start
    
    if hvd.rank() == 0:
        print(f'Elapsed time: {elapsed}')
        #np.save(os.path.join(model_dir, prefix + '_hist.p'), np.array(hist))


# In[ ]:



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ctc asr model')
    parser.add_argument('--filename', type=str,
                        help='filename of the model')
    parser.add_argument('--dataset', type=str,
                        help='path to the dataset index',)
    parser.add_argument('--val_dataset', type=str,
                        help='path to the validation dataset index (optional)',
                        default=None)
    parser.add_argument('--batch_size', type=int,
                        help='batch size for training and validation',
                        default=20)
    parser.add_argument('--lr', type=float,
                       help='initial learning rate',
                       default=0.005)
    parser.add_argument('--epochs', type=int,
                       help='number of epochs to use for training',
                       default=25)
    parser.add_argument('--tensorboard', type=bool,
                       help='if tensorboard log will be written',
                       default=False)
    parser.add_argument('--restart_filename', type=str,
                       help='filename of the checkpoint to restart from',
                       default=None)
    parser.add_argument('--mix_pres', type=bool,
                       help='if mixed precision training is requested',
                       default=False)
    args = parser.parse_args()

    train_model(filename=args.filename, dataset_idx=args.dataset,
                val_dataset_idx=args.val_dataset, epochs=args.epochs,
                batch_size=args.batch_size,
                tensorboard=args.tensorboard, restart_filename=args.restart_filename,
                is_mixed_precision=args.mix_pres, initial_lr=args.lr)







# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




