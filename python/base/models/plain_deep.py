from tensorflow import keras
from collections import namedtuple
from python.base.layers.simple_layer_params import *

import numpy as np
import tensorflow as tf
import logging
import math


ModelParams = namedtuple(
    'ModelParams', [
        'num_embedding_layers',
        'embedding_params',
        'num_dense_layers',
        'conv1d_params',
        'dense_params',
        'batchnorm_params',
        'dropout_params',
        'final_dense_params'
    ]
)

# assumes conv1d and dense layers are interleaved, thus # layers are different.
# may explore other interleaving architecture.
def constructModelParams(num_embedding_layers=1,
                         embedding_input_dims=[10],
                         embedding_output_dims=[32],
                         output_units=1,
                         conv1d_filters=8,
                         conv1d_filter_dim=11,
                         dense_shrink_multiplier=4):
  if (num_embedding_layers!=len(embedding_input_dims)) or (
      num_embedding_layers!=len(embedding_output_dims)):
      raise ValueError(f'length of embedding vocab_sizes and output dims must be equal to' +
                       f' num_embedding_layers: {num_embedding_layers} vs ' +
                       f'{len(embedding_input_dims)} vs {len(embedding_output_dims)}.')
  estimated_conv1d_output = np.sum(embedding_output_dims)*conv1d_filters
  num_dense_layers = int(math.log(estimated_conv1d_output/(output_units*dense_shrink_multiplier), dense_shrink_multiplier))
  logging.info(f'number of dense layers: {num_dense_layers}')
  embedding_params = []
  for i in range(num_embedding_layers):
    embedding_params.append(
        EmbeddingParams(embedding_input_dims[i], embedding_output_dims[i]))
  conv1d_params = []
  for i in range(num_dense_layers):
    conv1d_params.append(Conv1dParams(conv1d_filters, conv1d_filter_dim))
    conv1d_filter_dim -= 2
    conv1d_filter_dim = max(1, conv1d_filter_dim)

  dense_params = []
  dense_dim = estimated_conv1d_output
  for i in range(num_dense_layers):
    dense_dim = int(dense_dim/dense_shrink_multiplier)
    dense_dim = max(1, dense_dim)
    dense_params.append(DenseParams(dense_dim, 'relu'))
    logging.info(f'estimated {i}th dense layer dim: {dense_dim}')

  batchnorm_params = BatchnormParams(-1)
  dropout_params = DropoutParams(0.9)
  final_dense_params = DenseParams(output_units, 'relu')
  return ModelParams(num_embedding_layers, 
                     embedding_params, 
                     num_dense_layers, 
                     conv1d_params, 
                     dense_params, 
                     batchnorm_params, 
                     dropout_params,
                     final_dense_params)


# support param list for Embedding, Conv1D, Dense layers.
class PlainDeepNet(keras.Model):
    def __init__(self, 
                 string_lookups=None,
                 model_params=None):
        super().__init__()
         
        self.string_lookups = string_lookups
        self.model_params = model_params

        self.embedding_layers = []
        embedding_params = self.model_params.embedding_params
        for i in range(self.model_params.num_embedding_layers):
          self.embedding_layers.append(keras.layers.Embedding(
              input_dim = embedding_params[i].input_dim + 1,
              output_dim = embedding_params[i].output_dim,
              embeddings_initializer = embedding_params[i].embeddings_initializer,
              embeddings_regularizer = embedding_params[i].embeddings_regularizer,
              activity_regularizer = embedding_params[i].activity_regularizer,
              embeddings_constraint = embedding_params[i].embeddings_constraint,
              mask_zero = embedding_params[i].mask_zero,
              input_length = embedding_params[i].input_length,
              **embedding_params[i].kwargs
          ))
          logging.info(f'constructed the {i}th emebdding layer.')
        
        self.concatenate = keras.layers.Concatenate()

        self.conv1d_layers = []
        self.conv1d_normalization_layers = []
        conv1d_params = self.model_params.conv1d_params
        batchnorm_params = model_params.batchnorm_params
        for i in range(self.model_params.num_dense_layers):
            self.conv1d_layers.append(keras.layers.Conv1D(
                filters=conv1d_params[i].filters,
                kernel_size=conv1d_params[i].kernel_size,
                strides=conv1d_params[i].strides,
                padding=conv1d_params[i].padding,
                data_format=conv1d_params[i].data_format,
                dilation_rate=conv1d_params[i].dilation_rate,
                groups=conv1d_params[i].groups,
                activation=conv1d_params[i].activation,
                use_bias=conv1d_params[i].use_bias,
                kernel_initializer=conv1d_params[i].kernel_initializer,
                bias_initializer=conv1d_params[i].bias_initializer,
                kernel_regularizer=conv1d_params[i].kernel_regularizer,
                bias_regularizer=conv1d_params[i].bias_regularizer,
                activity_regularizer=conv1d_params[i].activity_regularizer,
                kernel_constraint=conv1d_params[i].kernel_constraint,
                bias_constraint=conv1d_params[i].bias_constraint,
                **conv1d_params[i].kwargs))
            self.conv1d_normalization_layers.append(
                keras.layers.Normalization(axis=batchnorm_params.axis))


        self.dense_layers = []
        self.dense_normalization_layers = []
        dense_params = self.model_params.dense_params
        for i in range(self.model_params.num_dense_layers):
            self.dense_layers.append(keras.layers.Dense(
                units=dense_params[i].units,
                activation=dense_params[i].activation,
                use_bias=dense_params[i].use_bias,
                kernel_initializer=dense_params[i].kernel_initializer,
                bias_initializer=dense_params[i].bias_initializer,
                kernel_regularizer=dense_params[i].kernel_regularizer,
                bias_regularizer=dense_params[i].bias_regularizer,
                activity_regularizer=dense_params[i].activity_regularizer,
                kernel_constraint=dense_params[i].kernel_constraint,
                bias_constraint=dense_params[i].bias_constraint,
                **dense_params[i].kwargs))
            self.dense_normalization_layers.append(
                keras.layers.Normalization(axis=batchnorm_params.axis))
            
        self.flatten = keras.layers.Flatten()
        '''
        # currently, batch normalization expects 3d data.
        self.normalization = keras.layers.BatchNormalization'''
        
        dropout_params = self.model_params.dropout_params
        self.dropout_layers = []
        for i in range(self.model_params.num_dense_layers):
            self.dropout_layers.append(
                keras.layers.Dropout(dropout_params.rate,
                                     noise_shape=dropout_params.noise_shape,
                                     **dropout_params.kwargs))
        
        dense_params = self.model_params.final_dense_params
        self.dense_final = keras.layers.Dense(units=dense_params.units,
                                              activation=dense_params.activation,
                                              use_bias=dense_params.use_bias,
                                              kernel_initializer=dense_params.kernel_initializer,
                                              bias_initializer=dense_params.bias_initializer,
                                              kernel_regularizer=dense_params.kernel_regularizer,
                                              bias_regularizer=dense_params.bias_regularizer,
                                              activity_regularizer=dense_params.activity_regularizer,
                                              kernel_constraint=dense_params.kernel_constraint,
                                              bias_constraint=dense_params.bias_constraint,
                                              **dense_params.kwargs)
        
        logging.info('construction completed.')
        
        self.softmax = tf.keras.layers.Softmax()

    def conv1d_module(self, i, inputs):
        outputs = self.conv1d_layers[i](inputs)
        outputs = self.conv1d_normalization_layers[i](outputs)
        return outputs
    
    def dense_module(self, i, inputs):
        outputs = self.dense_layers[i](inputs)
        outputs = self.dense_normalization_layers[i](outputs)
        outputs = self.dropout_layers[i](outputs)
        return outputs
    
    def call(self, inputs):
        embeddings = []
        for i in range(self.model_params.num_embedding_layers):
            output = self.string_lookups[i](inputs[i])
            embeddings.append(self.embedding_layers[i](output)) 
        concatenated_embeddings = self.concatenate(embeddings)
        concatenated_embeddings = tf.expand_dims(concatenated_embeddings, -1)
        logging.info(f'shape of concatenated embeddings: {concatenated_embeddings.shape}')
        outputs = concatenated_embeddings
        for i in range(self.model_params.num_dense_layers):
          outputs = self.conv1d_module(i, outputs)
          logging.info(f'output shape of {i}th conv1d layer: {outputs.shape}')
          outputs = self.flatten(outputs)
          outputs = self.dense_module(i, outputs)
          logging.info(f'output shape of {i}th dense layer: {outputs.shape}')
          if i != self.model_params.num_dense_layers-1:  
            outputs = tf.expand_dims(outputs, -1)
        
        '''model architect that separates conv1d and dense layers.
        for i in range(self.num_conv1d_layers):
            outputs = self.conv1d_module(outputs)
        logging.info(f'conv1d output shape: {outputs.shape}')
    
        # TODO: add the dense layer dim as a param.
        outputs = self.flatten(outputs)
        for i in range(self.num_dense_layers):
            outputs = self.dense_module(outputs)
        '''
        logging.info(f'shape of input to final dense layer: {outputs.shape}')

        outputs = self.dense_final(outputs)
        outputs = self.softmax(outputs)

        logging.info(f'final output shape: {outputs.shape}')

        return outputs