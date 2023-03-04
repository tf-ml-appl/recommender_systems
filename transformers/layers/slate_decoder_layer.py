from tensorflow import keras

from decoder_layer import *

class SlateDecoderLayer(keras.layers.Layer, keras.layers.DecoderLayer):
    '''A keras layer for transformer-based slate recommendation, used for decoding inpute items..'''
    def __init__(self):
        super(SlateDecoderLayer).__init__()