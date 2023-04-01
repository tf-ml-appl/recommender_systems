'a demo file to show the usage of MultiEmbeddings module and model.'

import sys
sys.path.insert(0, '/Users/YingyingZhu/Dropbox/ml_opensource/recommender_systems/python/base')

from layers.simple_layer_params import *
from models.multi_embedding_model import *
from gensim.parsing.preprocessing import remove_stopwords

import tensorflow as tf
import tensorflow_datasets as tfds

import logging


logging.getLogger().setLevel(logging.INFO)

# tunable model parameters.
BATCH_SIZE = 128*16
OUTPUT_UNITS = 5
EMBEDDING_SIZE = 32
MAX_FEATURE_SIZE = 3
MAX_WORDS = 200

# Ratings data.
ratings = tfds.load('movielens/100k-ratings', split="train")
# Features of all the available movies.
movies = tfds.load('movielens/100k-movies', split="train")

movie_title_features = ratings.map(lambda x: x['movie_title'])
targets = ratings.map(lambda x: x['user_rating'])

texts = []
for movie_title in movie_title_features.batch(BATCH_SIZE*10):
  texts += (list(map(lambda x: remove_stopwords(x.decode('utf-8').lower()), movie_title.numpy())))
  # encoded = tokenizer.tokenize(movie_title)
  logging.info(f'preprocessed {len(texts)} texts.')

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)
tokenized_texts = list(map(lambda x: x[:3] if len(x)>=3 else x + [MAX_WORDS]*(MAX_FEATURE_SIZE-len(x)), tokenizer.texts_to_sequences(texts)))

train_data = iter(tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(tokenized_texts), targets)).repeat().batch(BATCH_SIZE))

embeddings_params = []
for i in range(MAX_FEATURE_SIZE):
  embeddings_params.append(EmbeddingParams(MAX_WORDS+1, EMBEDDING_SIZE))

model = MultiEmbeddingModel(
    OUTPUT_UNITS,
    num_features=MAX_FEATURE_SIZE, 
    embeddings_params=embeddings_params)

# tunable
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    0.3, 500, 0.9, staircase=False, name=None
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.Accuracy()

def get_loss(y_pred, y, output_units=1):
  return tf.keras.losses.MeanSquaredError()(y_pred, y)

input_signature = [
    tf.TensorSpec(shape=(BATCH_SIZE, MAX_FEATURE_SIZE), dtype=tf.int32, name=None), 
    tf.TensorSpec(shape=(BATCH_SIZE, ), dtype=tf.float32, name=None)
]
@tf.function(input_signature=input_signature)
def train_step(inputs, targets):
  with tf.GradientTape() as tape:
    outputs = model(inputs, BATCH_SIZE)
    targets = tf.subtract(targets, 1)
    loss = loss_func(targets, outputs)
    predicts = tf.math.argmax(outputs, -1)
    metric.update_state(targets, predicts)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  result = {'loss': loss, 'metric': metric.result()}
  return result

for i in range(200):
  inputs, targets = train_data.get_next()
  result = train_step(inputs, targets)
  loss = result['loss'].numpy()
  metric = result['metric'].numpy()
  if i%100 == 0:
    logging.info(f'step {i} loss: {loss}')
    logging.info(f'metric: {metric}')