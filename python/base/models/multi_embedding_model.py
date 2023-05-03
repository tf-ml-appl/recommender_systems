'a demo plain deep model for multi-variant features'

from tensorflow import keras
from python.base.layers.simple_layer_params import *
from python.base.modules.multi_embeddings import *


class MultiEmbeddingModel(keras.Model):
  def __init__(self, output_units, num_features=1, embeddings_params=[EmbeddingParams()], embedding_merge_type='average'):
    super().__init__()

    self.multi_embedding_layer = MultiEmbeddings(
        num_features=num_features, 
        embeddings_params=embeddings_params)
    self.flatten = keras.layers.Flatten()
    # multiple dense layers may be added.
    self.dense = keras.layers.Dense(output_units)
    self.softmax = keras.layers.Softmax()

  def call(self, inputs, batch_size):
    outputs = self.multi_embedding_layer(inputs, batch_size)
    outputs = self.flatten(outputs)
    outputs = self.dense(outputs)
    outputs = self.softmax(outputs)
    return outputs