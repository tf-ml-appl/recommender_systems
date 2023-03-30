import sys
sys.path.insert(0, '/Users/YingyingZhu/Dropbox/ml_opensource/recommender_systems/python/base')
print(sys.path)
    
from collections import defaultdict
from tensorflow import keras
from models.plain_deep import *

import tensorflow_datasets as tfds
import tensorflow as tf

import logging

    
logging.getLogger().setLevel(logging.INFO)

BATCH_SIZE = 128*16
OUTPUT_UNITS = 5

# Ratings data.
ratings = tfds.load('movielens/100k-ratings', split="train")
# Features of all the available movies.
movies = tfds.load('movielens/100k-movies', split="train")

user_feature_keys= ['bucketized_user_age', 'raw_user_age', 'user_gender', 'user_id', 'user_occupation_label', 'user_occupation_text', 'user_zip_code']
movie_feature_keys = ['movie_genres', 'movie_id', 'movie_title']
extra_feature_keys = 'timestamp'
target = 'user_rating'

user_features = ratings.map(lambda x: {
    'bucketized_user_age': tf.strings.as_string(x['bucketized_user_age']),
    'raw_user_age': tf.strings.as_string(x['raw_user_age']),
    'user_gender': tf.strings.as_string(x['user_gender']),
    'user_id': x['user_id'],
    'user_occupation_label': tf.strings.as_string(x['user_occupation_label']),
    'user_occupation_text': x['user_occupation_text'],
    'user_zip_code': x['user_zip_code'],
})

movie_features = ratings.map(lambda x: {
    'movie_genres': tf.strings.as_string(tf.slice(x['movie_genres'], begin=[0], size=[1])),
    'movie_id': x['movie_id'],
    'movie_title': x['movie_title']
})

user_feature_vocabularies = defaultdict()
for feature in user_feature_keys:
  logging.info(f'processing feature {feature}.')
  user_feature_vocabularies[feature] = tf.keras.layers.StringLookup(mask_token=None)
  user_feature_vocabularies[feature].adapt(user_features.map(lambda x: x[feature] if x[feature].dtype==tf.string else tf.strings.as_string(x[feature])))

movie_feature_vocabularies = defaultdict()
for feature in movie_feature_keys:
  movie_feature_vocabularies[feature] = tf.keras.layers.StringLookup(mask_token=None)
  movie_feature_vocabularies[feature].adapt(movie_features.map(lambda x: x[feature] if x[feature].dtype==tf.string else tf.strings.as_string(x[feature])))

vocab_sizes = []
for vocab in user_feature_vocabularies.values():
  vocab_sizes.append(len(vocab.get_vocabulary()))

# add parse example

batched_user_features = ratings.map(lambda x: {
    'bucketized_user_age': tf.strings.as_string(x['bucketized_user_age']),
    'raw_user_age': tf.strings.as_string(x['raw_user_age']),
    'user_gender': tf.strings.as_string(x['user_gender']),
    'user_id': x['user_id'],
    'user_occupation_label': tf.strings.as_string(x['user_occupation_label']),
    'user_occupation_text': x['user_occupation_text'],
    'user_zip_code': x['user_zip_code'],
    'user_rating': x['user_rating']
}).prefetch(BATCH_SIZE*100).shuffle(BATCH_SIZE*50).repeat().batch(BATCH_SIZE)
dataset_iter = iter(batched_user_features)

num_embedding_layers=7
embedding_input_dims=vocab_sizes
embedding_output_dims=[32]*num_embedding_layers
output_units = OUTPUT_UNITS
conv1d_filters=8
conv1d_filter_dim=11
dense_shrink_multiplier=4

model_params = constructModelParams(num_embedding_layers=7,
                                    embedding_input_dims=embedding_input_dims,
                                    embedding_output_dims=embedding_output_dims,
                                    output_units=output_units,
                                    conv1d_filters=conv1d_filters,
                                    conv1d_filter_dim=conv1d_filter_dim,
                                    dense_shrink_multiplier=dense_shrink_multiplier)

model = PlainDeepNet(
    string_lookups=list(user_feature_vocabularies.values()),
    model_params=model_params)


# TODO: adds movie_feature_vocabularies
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    0.3, 500, 0.9, staircase=False, name=None
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.Accuracy()

def get_loss(y_pred, y, output_units=1):
  return tf.keras.losses.MeanSquaredError()(y_pred, y)

# y_one_hot = tf.one_hot(tf.cast(tf.math.subtract(inputs[-1], 1), dtype=tf.int32), OUTPUT_UNITS, dtype=tf.int32)
# y_one_hot.set_shape([BATCH_SIZE, OUTPUT_UNITS])  

input_signature = [[
    tf.TensorSpec(shape=(BATCH_SIZE, ), dtype=tf.string, name=None), 
    tf.TensorSpec(shape=(BATCH_SIZE, ), dtype=tf.string, name=None), 
    tf.TensorSpec(shape=(BATCH_SIZE, ), dtype=tf.string, name=None), 
    tf.TensorSpec(shape=(BATCH_SIZE, ), dtype=tf.string, name=None), 
    tf.TensorSpec(shape=(BATCH_SIZE, ), dtype=tf.string, name=None), 
    tf.TensorSpec(shape=(BATCH_SIZE, ), dtype=tf.string, name=None), 
    tf.TensorSpec(shape=(BATCH_SIZE, ), dtype=tf.string, name=None), 
    tf.TensorSpec(shape=(BATCH_SIZE, ), dtype=tf.float32, name=None)
]]
@tf.function(input_signature=input_signature)
def train_step(inputs):
  with tf.GradientTape() as tape:
    outputs = model(inputs[:-1])  
    targets = tf.subtract(inputs[-1], 1)
    loss = loss_func(targets, outputs)
    predicts = tf.math.argmax(outputs, -1)
    metric.update_state(targets, predicts)
    # loss = get_loss(outputs, inputs[-1], output_units=OUTPUT_UNITS)
    # print(model.trainable_variables)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  result = {'loss': loss, 'metric': metric.result()}
  # result = {'loss': loss, 'metric': metric.result(), 'targets': targets, 'predicts': predicts, 'outputs': outputs}
  return result

if __name__ == '__main__':
    # TODO: adds movie_feature_vocabularies
    for i in range(200):
        batch = dataset_iter.get_next()
        inputs = list(batch.values())
        # print(inputs[0].shape)
        result = train_step(inputs)
        loss = result['loss'].numpy()
        metric = result['metric'].numpy()
        if i%100 == 0:
            logging.info(f'step {i} loss: {loss}')
            logging.info(f'metric: {metric}')