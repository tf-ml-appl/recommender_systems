'a demo file to show the distributed training, with ps strategy and MultiEmbeddings module.'

# remove this after adding build file.
import sys
sys.path.insert(0, '/Users/user_name/faga-ml/recommender_systems/python/base')

import tensorflow as tf
import os
import multiprocessing
import portpicker
import sys
import logging

from layers.simple_layer_params import *
from modules.multi_embeddings import *
from tensorflow import keras
from gensim.parsing.preprocessing import remove_stopwords
from collections import namedtuple

import tensorflow as tf
import tensorflow_datasets as tfds

# from google.oauth2 import service_account
# from googleapiclient.discovery import build


# tunable model parameters.
BATCH_SIZE = 128*16
OUTPUT_UNITS = 5
EMBEDDING_SIZE = 32
MAX_FEATURE_SIZE = 3
MAX_WORDS = 200
FEATURE_PREPROCESS_BATCH_SIZE = BATCH_SIZE*10    

def create_in_process_cluster(num_workers, num_ps):
  """Creates and starts local servers and returns the cluster_resolver."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {}
  cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
  if num_ps > 0:
    cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]
  
  cluster_spec = tf.train.ClusterSpec(cluster_dict)

  # Workers need some inter_ops threads to work properly.
  worker_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1
  
  for i in range(num_workers):
    tf.distribute.Server(
        cluster_spec,
        job_name="worker",
        task_index=i,
        config=worker_config,
        protocol="grpc")

  for i in range(num_ps):
    tf.distribute.Server(
        cluster_spec,
        job_name="ps",
        task_index=i,
        protocol="grpc")
 
  cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, rpc_layer="grpc")
  return cluster_resolver

def data_preprocessing():
    # Ratings data.
    ratings = tfds.load('movielens/100k-ratings', split="train")

    movie_title_features = ratings.map(lambda x: x['movie_title'])
    texts = []
    for movie_title in movie_title_features.batch(FEATURE_PREPROCESS_BATCH_SIZE):
        texts += (list(map(lambda x: remove_stopwords(x.decode('utf-8').lower()), movie_title.numpy())))
        logging.info(f'preprocessed {len(texts)} texts.')
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(texts)
    tokenized_texts = list(map(lambda x: x[:3] if len(x)>=3 else x + [MAX_WORDS]*(MAX_FEATURE_SIZE-len(x)), tokenizer.texts_to_sequences(texts)))

    targets = ratings.map(lambda x: x['user_rating'])
    targets_list = []
    for target_list in targets.batch(FEATURE_PREPROCESS_BATCH_SIZE):
        targets_list += (list(map(lambda x: x.numpy(), target_list)))
        logging.info(f'preprocessed targets for {len(targets_list)} samples.')   
    
    return tokenized_texts, targets_list 
   
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARNING)

    os.environ["GRPC_FAIL_FAST"] = "use_caller"
    NUM_WORKERS = 2
    NUM_PS = 2
    cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)
    print(cluster_resolver.cluster_spec())
    
    variable_partitioner = (
        tf.distribute.experimental.partitioners.MinSizePartitioner(
            min_shard_bytes=(256 << 10),
            max_shards=1))

    strategy = tf.distribute.ParameterServerStrategy(
        cluster_resolver,
        variable_partitioner=variable_partitioner)   
    
    tokenized_texts, targets_list = data_preprocessing()
    
    def dataset_fn(_):
        train_data = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(tokenized_texts), 
                                          tf.data.Dataset.from_tensor_slices(targets_list))
                                         ).shuffle(FEATURE_PREPROCESS_BATCH_SIZE).batch(BATCH_SIZE).repeat()
        return train_data
    
    with strategy.scope():
        embeddings_params = []
        for i in range(MAX_FEATURE_SIZE):
            embeddings_params.append(EmbeddingParams(MAX_WORDS+1, EMBEDDING_SIZE))

        inputs = tf.keras.layers.Input(shape=(MAX_FEATURE_SIZE,), dtype=tf.int32, name='model_input')
        outputs = MultiEmbeddings(
            num_features=MAX_FEATURE_SIZE, 
            embeddings_params=embeddings_params)(inputs, BATCH_SIZE)
        outputs = keras.layers.Flatten()(outputs)
        outputs = keras.layers.Dense(OUTPUT_UNITS)(outputs)
        outputs = keras.layers.Softmax()(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # tunable
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            0.3, 500, 0.9, staircase=False, name=None
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
        metric = tf.keras.metrics.Accuracy()

    @tf.function
    def train_step(iterator):
        def replica_step(inputs, targets):
            with tf.GradientTape() as tape:  
                outputs = model(inputs, training=True)
                targets = tf.subtract(targets, 1)
                loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(targets, outputs)
                gradients = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            predicts = tf.math.argmax(outputs, -1)
            metric.update_state(targets, predicts)
            return loss
        
        inputs, targets = next(iterator) 

        losses = strategy.run(replica_step, args=(inputs, targets))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)
    
    coordinator = tf.distribute.coordinator.ClusterCoordinator(strategy)

    per_worker_dataset_fn = strategy.distribute_datasets_from_function(dataset_fn)
    per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
    per_worker_iterator = iter(per_worker_dataset) 

    num_epochs = 4
    steps_per_epoch = 10000

    for i in range(num_epochs):
        metric.reset_states()
        for _ in range(steps_per_epoch):
            coordinator.schedule(train_step, args=(per_worker_iterator,))
            coordinator.join()
        print("Finished epoch %d, accuracy is %f." % (i, metric.result().numpy()))
