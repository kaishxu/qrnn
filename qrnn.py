import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, GaussianNoise, Dropout
from keras.models import Model
from keras import regularizers
from tensorflow import keras
import tensorflow as tf

def qloss(y_true, y_pred):
    q = np.array(range(1, 100))
    left = (q / 100 - 1) * (y_true - y_pred)
    right = q / 100 * (y_true - y_pred)
    return K.mean(K.maximum(left, right))

def get_model(input_dim, num_units, act, dp, gauss_std=0.3, num_hidden_layers=1):
    input = Input((input_dim,), name='input')
    
    x = input
    
    for i in range(num_hidden_layers):
        x = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal',
                kernel_regularizer=regularizers.l2(0.001), activation=act[i])(x)
        x = Dropout(dp[i])(x)
        x = GaussianNoise(gauss_std[i])(x)  #似乎不适用与小模型?
    x = Dense(99, activation=None, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(x)

    model = Model(input, x)
    return model

class Qloss(keras.metrics.Metric):
    def __init__(self, name='qloss_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.qs = self.add_weight('qs', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        q = np.array(range(1, 100))
        left = (q / 100 - 1) * (y_true - y_pred)
        right = q / 100 * (y_true - y_pred)
        tmp = tf.reduce_mean(K.maximum(left, right))
        self.qs.assign_add(tmp)
        self.count.assign_add(1)
        
    def result(self):
        qs = self.qs / self.count
        return qs
