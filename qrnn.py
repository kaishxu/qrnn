import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model
from keras import regularizers

def qloss(y_true, y_pred):
    q = np.array(range(1, 100))
    tmp1 = (q / 100 - 1) * (y_true - y_pred)
    tmp2 = q / 100 * (y_true - y_pred)
    return K.mean(K.maximum(tmp1, tmp2))

def get_model(input_dim, num_units, act, gauss_std=0.3, num_hidden_layers=1):
    input_ = Input((input_dim,), name='input')
    
    x = input_
    
    for i in range(num_hidden_layers):
        x = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal', 
                kernel_regularizer=regularizers.l2(0.001), activation=act[i])(x)
        x = GaussianNoise(gauss_std)(x)  #似乎不适用与小模型?
    x = Dense(99, activation=None, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(x)

    model = Model(inputs=input_, outputs=x)
    model.compile(loss=qloss, optimizer='adam', metrics=['accuracy'])
    return model
