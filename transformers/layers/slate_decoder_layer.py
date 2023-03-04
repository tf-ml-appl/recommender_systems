from tensorflow import keras

class SlateDecoderLayer(keras.layers.Layer):
    '''A keras layer for transformer-based slate recommendation.'''
    def __init__(self):
        super(SlateDecoderLayer).__init__()