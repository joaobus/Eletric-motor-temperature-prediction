import numpy as np
import tensorflow as tf # type:ignore
from keras.layers import Input, LSTM, Dense, Flatten, add, Conv1D, MaxPool1D, SpatialDropout1D, GaussianNoise # type:ignore
from keras import models, optimizers as opts # type:ignore
from keras import regularizers # type:ignore
from utils.configs import rnn_rotor_cfg, rnn_stator_cfg, tcn_rotor_cfg, tcn_stator_cfg


def rnn_stator_model(n_features):
    n_targets = rnn_stator_cfg['n_out']
    n_units = rnn_stator_cfg['n_units']

    x = tf.keras.Input(shape=(rnn_stator_cfg['window'], n_features))
    x_before = x
    # layer 1
    y = LSTM(units=n_units,
             dropout=rnn_stator_cfg['dropout_rate'],
             kernel_regularizer=regularizers.L2(rnn_stator_cfg['reg_rate']))(x)
    y = GaussianNoise(rnn_stator_cfg['grad_noise'])(y)
    x_dense = Dense(n_units, activation='relu')(x_before)
    y = add([x_dense, y])

    # layer 2
    y_before = y
    y = LSTM(units=n_units,
             dropout=rnn_stator_cfg['dropout_rate'],
             kernel_regularizer=regularizers.L2(rnn_stator_cfg['reg_rate']))(y)
    y = GaussianNoise(rnn_stator_cfg['grad_noise'])(y)
    y = add([y_before, y])

    y = Flatten()(y)
    y = Dense(n_targets)(y)
    model = models.Model(outputs=y, inputs=x_before)
    return model


def rnn_rotor_model(n_features):
    n_targets = rnn_rotor_cfg['n_out']
    n_units = rnn_rotor_cfg['n_units']

    x = tf.keras.Input(shape=(rnn_rotor_cfg['window'], n_features))
    x_before = x
    y = LSTM(units=n_units,
             dropout=rnn_rotor_cfg['dropout_rate'],
             kernel_regularizer=regularizers.L2(rnn_rotor_cfg['reg_rate']))(x)
    y = GaussianNoise(rnn_rotor_cfg['grad_noise'])(y)
    x_dense = Dense(n_units, activation='relu')(x_before)
    y = add([x_dense, y])
    y = Flatten()(y)
    y = Dense(n_targets)(y)
    model = models.Model(outputs=y, inputs=x_before)
    return model


def cnn_stator_model(n_features):
    # input shape: window_size x len(x_cols)
    n_units = tcn_stator_cfg['n_units']
    l_kernel = tcn_stator_cfg['kernel_size']
    w = tcn_stator_cfg['window']
    x = Input((w, n_features))
    # tf.keras knows no causal padding :(
    # layer 1
    y = Conv1D(filters=n_units, kernel_size=l_kernel, activation='relu',
               padding='same')(x)
    # layer 2
    y = Conv1D(filters=n_units, kernel_size=l_kernel, activation='relu',
               padding='same', dilation_rate=2)(y)
    shortcut = Conv1D(filters=n_units, kernel_size=1, dilation_rate=2,
                      padding='same')(x)
    y = SpatialDropout1D(tcn_stator_cfg['dropout_rate'])(y)
    y = add([shortcut, y])

    shortcut = y
    # layer 3
    y = Conv1D(filters=n_units, kernel_size=l_kernel, activation='relu', padding='same')(x)
    # layer 4
    y = Conv1D(filters=n_units, kernel_size=l_kernel, activation='relu',
               padding='same', dilation_rate=4)(y)
    y = SpatialDropout1D(tcn_stator_cfg['dropout_rate'])(y)
    y = add([shortcut, y])

    y = MaxPool1D(pool_size=w)(y)
    y = Flatten()(y)
    y = Dense(units=tcn_stator_cfg['n_out'])(y)

    model = models.Model(inputs=x, outputs=y)
    return model


def cnn_rotor_model(n_features):
    # input shape: window_size x len(x_cols)
    n_units = tcn_rotor_cfg['n_units']
    l_kernel = tcn_rotor_cfg['kernel_size']
    x = Input((tcn_rotor_cfg['window'], n_features))
    y = Conv1D(filters=n_units, kernel_size=l_kernel, padding='same', activation='relu')(x)

    y = Conv1D(filters=n_units, kernel_size=l_kernel, padding='same',
                      dilation_rate=2, activation='relu')(y)

    shortcut = Conv1D(filters=n_units, kernel_size=1,
                             dilation_rate=2, padding='same')(x)
    y = SpatialDropout1D(tcn_stator_cfg['dropout_rate'])(y)
    y = add([shortcut, y])

    y = MaxPool1D(pool_size=tcn_rotor_cfg['window'])(y)
    y = Flatten()(y)
    y = Dense(units=tcn_rotor_cfg['n_out'])(y)

    model = models.Model(inputs=x, outputs=y)
    return model