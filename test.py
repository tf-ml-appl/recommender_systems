from tensorflow import keras
import tensorflow as tf
import random

BATCH_SIZE = 10
EMBEDDING_SIZE = 32
TOKEN_COUNT = 5000

class TestModel(tf.keras.Model):
    def __init__(self, batch_size, token_count, embedding_size):
        super().__init__()
        self.embedding_layer = keras.layers.Embedding(token_count, embedding_size, input_length=batch_size)
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(embedding_size, activation='relu')
        self.dense2 = keras.layers.Dense(1, activation='relu')

    def call(self, input, training=True):
        # training defaults to be True.
        output = self.embedding_layer(input, training=training)
        output = self.flatten(output, training=training)
        output = self.dense1(output, training=training)
        output = self.dense2(output, training=training)
        return output
    
def generate_dataset():
    random_list = tf.constant([0 for _ in range(TOKEN_COUNT)], shape=(TOKEN_COUNT, 1), dtype=tf.int32)
    # [random.randint(0, BATCH_SIZE-1) for _ in range(TOKEN_COUNT)]
    targets = tf.constant([random.randint(0,1) for i in range(TOKEN_COUNT)], shape=(TOKEN_COUNT, 1), dtype=tf.int32)
    data = tf.data.Dataset.from_tensor_slices(random_list)
    labels = tf.data.Dataset.from_tensor_slices(targets)
    dataset = tf.data.Dataset.zip((data, labels)).batch(BATCH_SIZE)
    return dataset

def generate_data():
    random_list = [1 for _ in range(TOKEN_COUNT)]
    targets = [random.randint(0,1) for i in range(TOKEN_COUNT)]    
    dataset = tf.data.Dataset.from_tensor_slices({'input': random_list, 'output': targets}).batch(BATCH_SIZE)
    iterator = iter(dataset)
    return iterator

def get_loss(pred_y, y):
    return loss_function(y, pred_y)
 
input_signature = [
    tf.TensorSpec(shape=(BATCH_SIZE, ), dtype=tf.int32),
    tf.TensorSpec(shape=(BATCH_SIZE, ), dtype=tf.int32)
]   

@tf.function(input_signature=input_signature)
def train_step(x, y):
    with tf.GradientTape() as tape:
        # print(f'trainable variables: {model.trainable_variables}')
        output = model(x)
        loss = get_loss(output, y)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    result = {'loss': loss}
    return result

# the training path is different from compile. in the other words, the eager execution does not support embedding updates.
if __name__=="__main__":
    '''
    model = TestModel(BATCH_SIZE, BATCH_SIZE, EMBEDDING_SIZE)
    optimizer = tf.keras.optimizers.Adam()
    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    # dataset = generate_dataset()
    # model.fit(dataset, epochs=1)
    # print(model.run_eagerly)
    # print(model.trainable_variables)
    dataset = generate_data()
    for i in range(3):
        batch = dataset.get_next()
        result = train_step(batch['input'], batch['output'])
        print(model.embedding_layer.get_weights()[0][0])
        print(f'step: {i}')
        print(result)
        # print(model.trainable_variables())
    '''
    '''
    a = tf.Variable([1,2,3], shape=(3,), dtype=tf.int32)
    b = tf.Variable([4,5,6], shape=(3,), dtype=tf.int32)
    c = tf.concat([a, b], 0)
    tf.compat.v1.assign(c, [3,4,5,6,7,8])
    print(a,b,c)
    '''
    input_shape = (4, 10, 128)
    x = tf.random.normal(input_shape)
    y = tf.keras.layers.Conv1D(
    32, 4, activation='relu',input_shape=input_shape[1:])(x)
    print(y.shape)
    print(tf.keras.layers.Conv1D(32, 4, activation='relu',input_shape=input_shape[1:]).variables)
    
        

        
    
            
            
            


    