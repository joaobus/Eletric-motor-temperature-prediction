import tensorflow as tf
import keras

from keras.layers import LSTM, GaussianNoise, Flatten, Dense, Add, Dropout
from keras.regularizers import L2



class RNNRegressor(keras.Model):
    def __init__(self, cfg):
        assert cfg['type'] == 'rnn', 'Invalid config, must be RNN'
        super().__init__()
        self.cfg = cfg
        self.lstm_return = LSTM(self.cfg['n_units'],
                                kernel_regularizer=L2(self.cfg['reg_rate']),
                                return_sequences=True)
        self.lstm_no_return = LSTM(self.cfg['n_units'],
                                   kernel_regularizer=L2(self.cfg['reg_rate']))
        self.grad_noise = GaussianNoise(self.cfg['grad_noise'])
        self.dropout = Dropout(self.cfg['dropout_rate'])
        self.flatten = Flatten()
        self.dense = Dense(self.cfg['n_out'], activation='linear')
        self.residual_add = Add()
    

    def call(self, x, training=False):
        for i in range(self.cfg['n_hidden']):
            inputs = x
            x = self.lstm_return(x) if i != (self.cfg['n_hidden'] - 1) else self.lstm_no_return(x) # Last layer doesn't return sequences
            if training:
                x = self.grad_noise(x)
                x = self.dropout(x)
            if i > 0:
                # Residual add can only be applied to the hidden layers so as to match shape 
                x = self.residual_add([x,inputs])
        x = self.flatten(x)
        return self.dense(x)
    
    

class TCNRegressor(keras.Model):
    def __init__(self, cfg):
        assert cfg['type'] == 'tcn', 'Invalid config, must be RNN'
        super().__init__()
        
    def call(self, x, training=False):
        pass