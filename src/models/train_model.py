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


def build_base_network(input_shape):
    inputs = Input(shape=input_shape)
    hidden = BatchNormalization()(inputs)
    # oups, forgot activation here... should have been:
    # Dense(16, activation='relu')
    hidden = Dense(16)(hidden)
    hidden = BatchNormalization()(hidden)
    model = Model(inputs=inputs, outputs=hidden)
    return model

def create_mdn_estimator(X, y, X_val, y_val, batch_size=32, nb_epochs=1000, nb_mixtures=1):
    input_shape = X.shape[1:]
    base_network = build_base_network(input_shape)

    input_a = Input(shape=input_shape)
    x = base_network(input_a)
    mixture_layer = mdn.MDN(1, nb_mixtures)(x)
    mdn_model = Model([input_a], [mixture_layer])

    loss = mdn.get_mixture_loss_func(1, nb_mixtures)
    mdn_model.compile(loss=loss, optimizer='adam')

    tb_cb = TensorBoard(log_dir='./mdn_logs/{}'.format(time.time()),
                        histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                        embeddings_data=None, update_freq='epoch')

    mdn_model.fit(
        X,
        y,
        batch_size=batch_size,
        epochs=nb_epochs,
        validation_data=(X_val, y_val),
        verbose=1, callbacks=[tb_cb])

    return mdn_model

def make_mdn_prediction(mdn_model, X_test, return_sample=False, nb_mixture=1, scaling=1.):
    assert nb_mixture == 1, "Only 1 mixture is supported"

    y_pred = mdn_model.predict(X_test)
    y_mu = y_pred[:, 0] * scaling
    y_std = np.sqrt(y_pred[:, 1] * scaling ** 2)
    if return_sample:
        y_sample = np.apply_along_axis(sample_from_output, 1, y_pred, 1, nb_mixture, temp=1.0)
        return y_mu, y_std, y_sample
    else:
        return y_mu, y_std

def main():
    load_dotenv(find_dotenv())
    input_data_file = Path(os.environ["project_dir"]) / "data/external/pcm_main_rotor_embeded.npz"
    # input_data_file = Path(os.environ["project_dir"]) / "data/external/pcm_main_rotor.npz"
    data = np.load(input_data_file)
    X, y = data["X"], data["y"]

    nb_epochs = 1000
    batch_size = 32
    nb_mixtures = 1

    X = normalize(X, axis=0, norm="max")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    max_y_train = np.max(y_train)
    y_train /= max_y_train
    X_val = X_test
    y_val =  y_test / max_y_train

    model = create_mdn_estimator(X_train, y_train, X_val, y_val, batch_size=batch_size, nb_epochs=nb_epochs, nb_mixtures=nb_mixtures)

    y_mu_test, y_std_test = make_mdn_prediction(model, X_test, scaling=max_y_train)
    y_true = y_test
    n_to_show = 50
    plt.scatter(np.arange(y_true.shape[0])[:n_to_show], np.squeeze(y_true)[:n_to_show], label="Actual observation")
    plt.errorbar(np.arange(y_true.shape[0])[:n_to_show], np.squeeze(y_mu_test)[:n_to_show], fmt='go', marker="o", ecolor="g", yerr=np.squeeze(y_std_test)[:n_to_show], label="Predicted mean and std")
    plt.legend()
    plt.title("Prediction using PCM data")
    plt.xlabel("Manoeuvers (arbitrary indexing)")
    plt.ylabel("Load")
    plt.show()
    print(np.mean(np.square(y_true - y_mu_test)))
    print(np.linalg.norm(y_true - y_mu_test) / np.linalg.norm(y_true))
    print(np.max(y_true), np.max(y_mu_test), np.max(np.abs(y_true - y_mu_test)))
    print(np.mean(y_std_test))


if __name__ == "__main__":
    main()
