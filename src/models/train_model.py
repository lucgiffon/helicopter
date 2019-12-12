import os
import time
from pathlib import Path

import tensorflow as tf
from dotenv import load_dotenv, find_dotenv
from keras.callbacks import TensorBoard
from mdn import sample_from_output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from keras.layers import Dense, Input, Concatenate, BatchNormalization
from keras.models import Model
import numpy as np
from sklearn import preprocessing
import keras.backend as K
import mdn
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------- Config params

def learn_distribution(X, y):
    """

    :param X:
    :param y:
    :return: estimator
    """

    model = build_model(
        loss=lambda y, f: gnll_loss(y, f),
        input_shape=X.shape[1:],
        n_neurons=16)

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=n_epochs,
        validation_data=(X_test, y_test),
        verbose=1)

    return model


def gnll_loss(y, f):
    """ Computes the negative log-likelihood loss of y given the parameters.
    """
    mu = f[:, 0]
    sigma = f[:, 1]
    gaussian_dist = tf.distributions.Normal(loc=mu, scale=sigma)
    log_prob = gaussian_dist.log_prob(value=y)
    neg_log_likelihood = -1.0 * K.sum(log_prob)
    return neg_log_likelihood

def build_model(loss, input_shape, n_neurons=16):
    """builds the density network
    """
    inputs = Input(shape=input_shape)
    hidden = Dense(n_neurons)(inputs)
    mixture_layer = mdn.MDN(1, 1)(hidden)
    loss = mdn.get_mixture_loss_func(1, 1)
    model = Model(inputs=inputs, outputs=mixture_layer)
    model.compile(loss=loss, optimizer='nadam')
    return model


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    input_data_file = Path(os.environ["project_dir"]) / "data/external/pcm_main_rotor_embeded.npz"
    # input_data_file = Path(os.environ["project_dir"]) / "data/external/pcm_main_rotor.npz"
    data = np.load(input_data_file)
    X, y = data["X"], data["y"]

    input_shape = X.shape[1:]
    n_epochs = 1000
    batch_size = 32
    N_MIXES = 1

    X = normalize(X, axis=0, norm="max")
    # y = normalize(y[:, np.newaxis], axis=0, norm="max")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    max_y_train = np.max(y_train)
    y_train /= max_y_train
    y_test /= max_y_train
    # norm_y_train = np.linalg.norm(y_train)
    # y_train /= norm_y_train
    # y_test /= norm_y_train

    inputs = Input(shape=input_shape)
    hidden = BatchNormalization()(inputs)
    hidden = Dense(16)(hidden)
    hidden = BatchNormalization()(hidden)
    mixture_layer = mdn.MDN(1, N_MIXES)(hidden)
    loss = mdn.get_mixture_loss_func(1, N_MIXES)
    model = Model(inputs=inputs, outputs=mixture_layer)
    model.compile(loss=loss, optimizer='adam')


    tb_cb = TensorBoard(log_dir='./mdn_logs/{}'.format(time.time()),
                histogram_freq=0, batch_size=32, write_graph=True, write_grads=True , write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=n_epochs,
        validation_data=(X_test, y_test),
        verbose=1, callbacks=[tb_cb])

    y_pred = model.predict(X_test)
    # y_samples = np.apply_along_axis(sample_from_output, 1, y_pred, 1, N_MIXES, temp=1.0)
    y_samples = y_pred[:, 0]
    y_std = np.sqrt(y_pred[:, 1] * max_y_train**2)
    y_test *= max_y_train
    y_samples *= max_y_train
    print(np.mean(np.square(y_test-y_samples)))
    print(np.linalg.norm(y_test - y_samples)/np.linalg.norm(y_test))
    print(np.max(y_test), np.max(y_samples), np.max(np.abs(y_test-y_samples)))
    n_to_show = 100
    plt.scatter(np.arange(y_test.shape[0])[:n_to_show], np.squeeze(y_test)[:n_to_show], label="True")
    plt.errorbar(np.arange(y_test.shape[0])[:n_to_show], np.squeeze(y_samples)[:n_to_show], fmt='go', marker="o", ecolor="g", yerr=np.squeeze(y_std)[:n_to_show], label="Predicted")
    plt.legend()
    plt.show()

    # learn_distribution(X_train, y_train)


    y_pred = model.predict(X_train)
    # y_samples = np.apply_along_axis(sample_from_output, 1, y_pred, 1, N_MIXES, temp=1.0)
    y_samples = y_pred[:, 0]
    y_std = np.sqrt(y_pred[:, 1] * max_y_train**2)
    y_train *= max_y_train
    y_samples *= max_y_train
    print(np.mean(np.square(y_train-y_samples)))
    print(np.linalg.norm(y_train - y_samples)/np.linalg.norm(y_train))
    print(np.max(y_train), np.max(y_samples), np.max(np.abs(y_train-y_samples)))
    n_to_show = 50
    plt.scatter(np.arange(y_train.shape[0])[:n_to_show], np.squeeze(y_train)[:n_to_show], label="True")
    plt.errorbar(np.arange(y_train.shape[0])[:n_to_show], np.squeeze(y_samples)[:n_to_show], fmt='go', marker="o", ecolor="g", yerr=np.squeeze(y_std)[:n_to_show], label="Predicted")
    plt.legend()
    plt.show()

    # learn_distribution(X_train, y_train)

