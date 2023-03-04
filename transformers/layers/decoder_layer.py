from tensorflow import keras

class DecoderLayer(keras.layers.Layer):
    '''A keras layer for transformer-based recommendation, used for decoding inpute items..'''
    def __init__(self):
        super(DecoderLayer).__init__()