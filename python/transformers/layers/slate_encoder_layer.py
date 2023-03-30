from tensorflow import keras

from encoder_layer import *

class SlateEncoderLayer(keras.layers.Layer, keras.layers.EncoderLayer):
    '''A keras layer for transformer-based slate recommendation, used for encoding inpute items.'''
    def __init__(self):
        super(SlateEncoderLayer).__init__()
        

    