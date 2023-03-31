from layers.simple_layer_params import *

import tensorflow as tf


# assumes the inputs are tokenized and mapped to (0,...,token_counts) for each features.
# currently, text preprocessing (e.g. tokenizer) handles raw texts. 
# Used a third-party text preprocessing for extra features
# may add keras tokenizer here after supports tensor inputs.
class MultiEmbeddings(tf.keras.Module):
  def __init__(self, num_features=1, embeddings_params=[EmbeddingParams()]):
    super().__init__()
    if num_features != len(embeddings_params):
      raise ValueError('number of features must be equal to the number of embedding params: {num_feature} vs {len(embeddings_params)}.')
    self.num_feature = num_feature
    self.embedding_layers = []
    for i in range(num_features,
                   embeddings_params=embeddings_params,
                   merge_type='average'):
      self.embedding_layers.append(tf.keras.Embedding(
          input_dim = embeddings_params[i].input_dim + 1,
          output_dim = embeddings_params[i].output_dim,
          embeddings_initializer = embeddings_params[i].embeddings_initializer,
          embeddings_regularizer = embeddings_params[i].embeddings_regularizer,
          activity_regularizer = embeddings_params[i].activity_regularizer,
          embeddings_constraint = embeddings_params[i].embeddings_constraint,
          mask_zero = embeddings_params[i].mask_zero,
          input_length = embeddings_params[i].input_length,
          **embeddings_params[i].kwargs
      ))

      # assign a merge layer to merge the embeddings per example, based on merge_type
      
      
    def call(self, inputs):
        outputs = []
        for i in range(self.num_feaure):
            outputs.append(self.embedding_layers(inputs[i]))
        outputs = self.merge(outputs)
        return outputs
            
      