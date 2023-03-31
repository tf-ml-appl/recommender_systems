from layers.simple_layer_params import *

import tensorflow as tf

# TODO, may be supports features of variant-length (the computation burden would be large due to insonsistent feature sizes.
# check if all types of merge layers support multiple features larger than two.

# assumes the inputs are tokenized and mapped to (0,...,token_counts) for each features.
# currently, text preprocessing (e.g. tokenizer) handles raw texts. 
# Used a third-party text preprocessing for extra features
# may add keras tokenizer here after supports tensor inputs.
class MultiEmbeddings(tf.keras.Module):
  def __init__(self, num_features=1, embeddings_params=[EmbeddingParams()], merge_type='average'):
    super().__init__()
    if num_features != len(embeddings_params):
      raise ValueError('number of features must be equal to the number of embedding params: {num_feature} vs {len(embeddings_params)}.')
    self.num_feature = num_features
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

      # assigns a merge layer to merge the embeddings per example, based on merge_type
      if merge_type == 'concatenate':
        self.merge = tf.keras.layers.Concatenate()
      elif merge_type == 'average':
        self.merge = tf.keras.Average()
      elif merge_type == 'maximum':
        self.merge = tf.keras.Maximum()
      elif merge_type == 'minimum':
        self.merge = tf.keras.Minimum()
      elif merge_type == 'add':
        self.merge = tf.keras.Add()
      elif merge_type == 'subtract':
        self.merge = tf.keras.Subtract()
      elif merge_type == 'multiply':
        self.merge = tf.keras.Multiply()
      elif merge_type == 'dot':
        self.merge = tf.keras.Dot()
      else:
        raise ValueError('invalid merge type: {merge_type}, must be one of concatenate, average, maximum, minimum, add, subtract, multiply and dot')
      
    def call(self, inputs):
        outputs = []
        for i in range(self.num_feaure):
            outputs.append(self.embedding_layers(inputs[i]))
        outputs = self.merge(outputs)
        return outputs
            