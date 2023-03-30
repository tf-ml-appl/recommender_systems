from tensorflow import keras

class PlainDeepNet(keras.Model):
    def __init__(self, 
                 conv1d_params=None,
                 dense_params=None,
                 batchnorm_params=None,
                 dropout_params=None,
                 num_conv1d=1, 
                 num_dense=1):
        super(PlainDeepNet).__init__()
        
        self.conv1d = keras.layers.Conv1D(filters=conv1d_params.filters,
                                          kernel_size=conv1d_params.kernel_size,
                                          strides=conv1d_params.strides,
                                          padding=conv1d_params.padding,
                                          data_format=conv1d_params.data_format,
                                          dilation_rate=conv1d_params.dilation_rate,
                                          groups=conv1d_params.groups,
                                          activation=conv1d_params.activation,
                                          use_bias=conv1d_params.use_bias,
                                          kernel_initializer=conv1d_params.kernel_initializer,
                                          bias_initializer=conv1d_params.bias_initializer,
                                          activity_regularizer=conv1d_params.activity_regularizer,
                                          kernel_constraint=conv1d_params.kernel_constraint,
                                          bias_constraint=conv1d_params.bias_constraint,
                                          **conv1d_params.kwargs
        )
        self.dense = keras.layers.Dense(units=dense_params.units,
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
        
        self.flatten = keras.layers.Flatten()
        self.normalization = keras.layers.BatchNormalization(axis=batchnorm_params.axis,
                                                             momentum=batchnorm_params.momentum,
                                                             epsilon=batchnorm_params.epsilon,
                                                             center=batchnorm_params.center,
                                                             scale=batchnorm_params.scale,
                                                             beta_initializer=batchnorm_params.beta_initializer,
                                                             gamma_initializer=batchnorm_params.gamma_initializer,
                                                             moving_mean_initializer=batchnorm_params.moving_mean_initializer,
                                                             beta_regularizer=batchnorm_params.beta_regularizer,
                                                             gamma_regularizer=batchnorm_params.gamma_regularizer,
                                                             beta_constraint=batchnorm_params.beta_constraint,
                                                             gamma_constraint=batchnorm_params.gamma_constraint,
                                                             **batchnorm_params.kwargs)
        
        self.dropout = keras.layers.DropOut(dropout_params.rate,
                                            noise_shape=dropout_params.noise_shape,
                                            seed=dropout_params.seed,
                                            **dropout_params.kwargs
                                            )
        
    def conv1d_module(self, inputs):
        outputs = self.conv1d(inputs)
        outputs = self.normalization(outputs)
        return outputs
    
    def dense_module(self, inputs):
        outputs = self.dense(inputs)
        outputs = self.normalization(outputs)
        outputs = self.dropout(outputs)
    
    def call(self, inputs):
        embeddings = []
        for i in range(len(inputs)):
            embeddings.append(self.embedding_layers[i](inputs[i])) 
        concatenated_embeddings = keras.layers.Concatenate(embeddings)
        for i in range(self.num_conv1d):
            concatenated_embeddings = self.conv1d_module(concatenated_embeddings)
        output = concatenated_embeddings
        # TODO: add the dense layer dim as a param.
        for i in range(self.num_dense):
            output = self.dense_module(output)
        output = self.dense_final(output)
        return output
        
        