import tensorflow as tf
import keras

from keras.layers import LSTM, GaussianNoise, Dense, Add, Dropout, Conv1D, MaxPool1D, GlobalAveragePooling1D
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
        self.avg_pool = GlobalAveragePooling1D()
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
        x = self.avg_pool(x)
        return self.dense(x)
    
    

class TCNRegressor(keras.Model):
    def __init__(self, cfg):
        assert cfg['type'] == 'tcn', 'Invalid config, must be RNN'
        super().__init__()

        self.cfg = cfg
        self.dilation_rates = [2**i for i in range(self.cfg['n_hidden'])]
        self.conv_layers = [Conv1D(filters=self.cfg['n_units'], kernel_size=self.cfg['kernel_size'], 
                                   activation='relu', padding='same', 
                                   dilation_rate=i) for i in self.dilation_rates]
        self.dropout = Dropout(self.cfg['dropout_rate'])
        self.max_pooling = MaxPool1D(pool_size=self.cfg['window'])
        self.avg_pool = GlobalAveragePooling1D()
        self.dense = Dense(self.cfg['n_out'], activation='linear')
        self.residual_add = Add()

        
    def call(self, x, training=False):
        for i in range(self.cfg['n_hidden']):
            inputs = x
            x = self.conv_layers[i](x)
            if training:
                x = self.dropout(x)
            if i > 0:
                x = self.residual_add([x, inputs])
        x = self.max_pooling(x)
        x = self.avg_pool(x)
        return self.dense(x)