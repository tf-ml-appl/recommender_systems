from tensorflow import keras

class EncoderLayer(keras.layers.Layer):
    '''A keras layer for transformer-based recommendation, used for encoding inpute items..'''
    def __init__(self):
        super(EncoderLayer).__init__()
        
