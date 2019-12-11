import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate, Activation
from tensorflow.keras.models import Model
import numpy as np
import scikit as sklearn
from sklearn import preprocessing

np.random.seed(42)

# ---------------- Config params

n_epochs = 1000
batch_size = 128
train_part = 0.8

n_train = int(train_part*len(Y))
ix = np.random.permutation(len(Y))
ix_train = ix[:n_train]
ix_train = np.array(sorted(ix_train))
ix_test = ix[n_train:]
ix_test = np.array(sorted(ix_test))
print(len(ix_test))

X_train = X[ix_train,:]

y_train = Y[ix_train]
print(y_train.shape)
X_test = X[ix_test,:]

y_test = Y[ix_test]
print(y_test.shape)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
num_inputs = X_train.shape[1]



def learn_distribution(X, y):
    """

    :param X:
    :param y:
    :return: estimator
    """
    tf.keras.utils.get_custom_objects().update({'nnelu': Activation(nnelu)})

    model = build_model(
            loss=lambda y,
            f: gnll_loss(y,f), 
            num_inputs = num_inputs,
            n_neurons = 128)
    
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=n_epochs, 
        validation_data=(X_test, y_test),
        verbose=0)

    return model


def gnll_loss(y, f):
    """ Computes the negative log-likelihood loss of y given the parameters.
    """
    mu = f[:, 0]
    sigma = f[:, 1]
    gaussian_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma)
    log_prob = gaussian_dist.log_prob(value=y)
    neg_log_likelihood = -1.0 * tf.reduce_sum(log_prob)
    return neg_log_likelihood


def nnelu(input):
    """ Computes the Non-Negative Exponential Linear Unit
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))

def build_model(loss="mse", num_inputs= 1, num_outputs=2, n_neurons = 16):
    """builds the density network
    """
    inputs = Input(shape=(num_inputs,))
    hidden = Dense(n_neurons)(inputs)
    sigma = Dense(1, activation='nnelu')(hidden)
    mu = Dense(1, activation='linear')(hidden)
    out = Concatenate(name="output")([mu,sigma])
    model = Model(inputs=inputs, outputs=out)
    model.compile(loss=loss, optimizer='nadam')
    return model
