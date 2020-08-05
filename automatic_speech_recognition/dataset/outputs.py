import tensorflow as tf
from tensorflow import keras

class ModelOutputsDataset(keras.utils.Sequence):
    def __init__(self, pipeline, dataset, device='gpu:0', cache=False):
        # To wrap dataset in feature extractor it is required to create pipiline object
        self._feature_dataset = pipeline.wrap_preprocess(dataset)
        self._model = pipeline.model
        self._device = device
        
        self.should_cache = cache
        if self.should_cache:
            self._cache = {}
    
    def __len__(self):
        return len(self._feature_dataset)
    
    def __getitem__(self, i):
        if not self.should_cache:
            return self.get_batch(i)
        else:
            return self.get_batch_cache_on(i)
        
    def get_batch(self, i):
        with tf.device(self._device):
            X, y = self._feature_dataset[i]
            return X, self._model(X)
        
    def get_batch_cache_on(self, i):
        assert self.should_cache, "Caching must be turned on to use caching"
        
        if i not in self._cache:
            X, y = self.get_batch(i)
            self._cache[i] = self.get_batch(i)
        return self._cache[i]
    
    def precompute_all(self):
        assert self.should_cache, "Caching must be turned on to cache all dataset"
        
        for i in tqdm(range(len(self))):
            self.get_batch_cache_on(i)
        del self._model
        