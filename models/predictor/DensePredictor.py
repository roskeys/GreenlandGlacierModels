from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.activations import tanh


def getDensePredictor(target_shape=1, name='Dense'):
    input_array = Input(shape=(256,))
    x = Dense(128)(input_array)
    x = LeakyReLU()(x)
    x = Dense(64, activation=tanh)(x)
    pred = Dense(target_shape)(x)
    m = Model(inputs=input_array, outputs=pred, name=name)
    return m