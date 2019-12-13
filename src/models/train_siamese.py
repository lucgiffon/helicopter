import time

from dotenv import load_dotenv, find_dotenv
import os
import numpy as np
from pathlib import Path

import tensorflow as tf

from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Lambda, BatchNormalization
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import keras.backend as K


def create_pairs(x, y):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            obs_i, obs_j = x[i], x[j]
            pairs += [[obs_i, obs_j]]
            labels += [(y[i] - y[j])**2]
    return np.array(pairs), np.array(labels)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Dense(30, activation='relu')(input)
    x = BatchNormalization()(x)
    # x = Dropout(0.1)(x)
    return Model(input, x)

def create_siamese_embedder(X, y, X_val, y_val, batch_size=128, nb_epochs=10):
    input_shape = X.shape[1:]

    train_pairs, train_y = create_pairs(X, y)
    test_pairs, test_y = create_pairs(X_val, y_val)

    base_network = create_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    siamese_model = Model([input_a, input_b], distance)

    tb_cb = TensorBoard(log_dir='./logs_siamese/{}'.format(time.time()),
                        histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                        embeddings_data=None, update_freq='batch')

    # train
    adam = Adam(1e-1)
    siamese_model.compile(loss="mse", optimizer=adam)
    siamese_model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_y,
              batch_size=batch_size,
              epochs=nb_epochs,
              validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_y),
              callbacks=[tb_cb])


    transformation_model = Model(input=[input_a], output=[processed_a])
    transformation_model.compile(optimizer=adam, loss="mse")
    return transformation_model

def main():
    load_dotenv(find_dotenv())
    input_data_file = Path(os.environ["project_dir"]) / "data/external/pcm_main_rotor.npz"
    data = np.load(input_data_file)
    X, y = data["X"], data["y"]
    batch_size=128
    nb_epochs = 5

    X = normalize(X, axis=0, norm="max")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    max_y_train = np.max(y_train)
    y_train /= max_y_train
    y_test /= max_y_train

    transformation_model = create_siamese_embedder(X_train, y_train, X_test, y_test, batch_size=batch_size, nb_epochs=nb_epochs)
    transformed_data = transformation_model.predict(X)

    output_data_file = Path(os.environ["project_dir"]) / "data/external/pcm_main_rotor_embeded.npz"
    np.savez(output_data_file, **{"X": transformed_data, "y": y})


if __name__ == "__main__":
    main()