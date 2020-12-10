import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, GaussianNoise, Dropout
from keras.models import Model
from keras import regularizers
from tensorflow import keras
import tensorflow as tf

def qloss(y_true, y_pred, n_q=99):
    q = np.array(range(1, n_q + 1))
    left = (q / (n_q + 1) - 1) * (y_true - y_pred)
    right = q / (n_q + 1) * (y_true - y_pred)
    return K.mean(K.maximum(left, right))

def get_model(input_dim, num_units, act, dp=0.1, gauss_std=0.3, num_hidden_layers=1, num_quantiles=99):
    input = Input((input_dim,), name='input')
    
    x = input
    
    for i in range(num_hidden_layers):
        x = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal',
                kernel_regularizer=regularizers.l2(0.001), activation=act[i])(x)
        x = Dropout(dp[i])(x)
        x = GaussianNoise(gauss_std[i])(x)  #似乎不适用与小模型?
    
    x = Dense(num_quantiles, activation=None, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(x)

    model = Model(input, x)
    return model
