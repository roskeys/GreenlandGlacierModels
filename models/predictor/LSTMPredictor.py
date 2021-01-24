from tensorflow import expand_dims
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import tanh


def getLSTMPredictor(target_shape=1, name='LSTM'):
    input_array = Input(shape=(256,))
    x = expand_dims(input_array, -1)
    x = Dense(64, activation=tanh)(x)
    pred = Dense(target_shape)(x)
    m = Model(inputs=input_array, outputs=pred, name=name)
    return m
