from tensorflow.keras.layers import Embedding
import tensorflow.io.logging as logging

@keras_export('keras.layers.UniEmbedding')
class UniEmbedding(Embedding):
    
    def __init__(self,
                 input_dim=None,
                 output_dim=None,
                 embeddings_initializer='uniform',
                 embedding_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 mutable=True,
                 vocabulary_file=None,
                 **kwargs):
        super(UniEmbedding).__init__(input_dim,
                                     output_dim,
                                     embeddings_initializer,
                                     embedding_regularizer,
                                     activity_regularizer,
                                     embeddings_constraint,
                                     mask_zero,
                                     input_length,
                                     kwargs)
        self._mutable = mutable
        self._vocabulary_file = vocabulary_file
            
        def update_vocabulary(self, step, deleted_token_ids, extra_token_counts):
            '''updates the embedding variables for the layer to include new vocabularies. 
            The update of vocabulary file (e.g. vocabs with mapping) should be handled before calling this function.'''
            if not self._mutable:
                logging.Warning('the uni_embedding layer is not mutable.')
                return 
            # use initializer, add new variable/weight, delete the old one or reuse the name?
            
            
            
            
            
            
            
            
       

            