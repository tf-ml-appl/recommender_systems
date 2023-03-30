from collections import namedtuple

EmbeddingParams = namedtuple(
    'EmbeddingParams', [
        'input_dim',
        'output_dim',
        'embeddings_initializer',
        'embeddings_regularizer',
        'activity_regularizer',
        'embeddings_constraint',
        'mask_zero',
        'input_length',
        'sparse', # not used, not recognized in tensorflow 2.11.0
        'kwargs'],
        defaults=[10, 32, 'uniform', None, None, None, False, None, False, {}]
)

Conv1dParams = namedtuple(
    'Conv1dParams', [
        'filters', 
        'kernel_size', 
        'strides', 
        'padding', 
        'data_format', 
        'dilation_rate',
        'groups', 
        'activation', 
        'use_bias', 
        'kernel_initializer', 
        'bias_initializer',
        'kernel_regularizer',
        'bias_regularizer',
        'activity_regularizer', 
        'kernel_constraint',
        'bias_constraint',
        'kwargs'],
        defaults=[1, 5, 1, 'valid', 'channels_last', 1, 1, None, True, 'glorot_uniform', 'zeros', None, None, None, None, None, {}])

DenseParams = namedtuple(
    'DenseParams', [
        'units', 
        'activation', 
        'use_bias', 
        'kernel_initializer',
        'bias_initializer', 
        'kernel_regularizer',
        'bias_regularizer',
        'activity_regularizer', 
        'kernel_constraint', 
        'bias_constraint',
        'kwargs'],
        defaults=[10, None, True, 'glorot_uniform', 'zeros', None, None, None, None, None, {}])

BatchnormParams = namedtuple(
    'BatchnormParams', [
        'axis', 
        'momentum', 
        'epsilon', 
        'center', 
        'scale', 
        'beta_initializer', 
        'gamma_initializer', 
        'moving_mean_initializer', 
        'moving_variance_initializer',
        'beta_regularizer',
        'gamma_regularizer', 
        'beta_constraint', 
        'gamma_constraint',
        'synchronized', # not used, cannot found in tf-2.11.0
        'kwargs'],
        defaults=[1, 0.99, 0.001, True, True, 'zeros', 'ones', 'zeros', 'ones', None, None, None, None, False, {}])                                                          

DropoutParams = namedtuple(
    'DropoutParams', [
        'rate', 
        'noise_shape', 
        'seed',
        'kwargs'],
        defaults=[0.8, None, None, {}])
