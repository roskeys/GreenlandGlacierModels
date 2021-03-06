from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.activations import tanh, relu
from models.components.common import getInput, flattenAll, getOutput, LeakyReLU


def getModel(cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim,
             ocean_dim=None, other_dim=None, target_shape=1, name="ann"):
    # input to the model
    input_array = getInput(cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim,
                           ocean_dim)
    if other_dim:
        other_in = Input(shape=other_dim, name="OtherParam")
        input_array.append(other_in)
    # flatten all data and concatenate together
    x = flattenAll(input_array)
    x = BatchNormalization()(x)
    x = Dense(512, activation=relu)(x)
    x = BatchNormalization()(x)

    # unify output layer
    x = Dense(256, activation=relu)(x)
    x = BatchNormalization()(x)
    # last stage processing
    x = LeakyReLU()(Dense(128)(x))
    x = BatchNormalization()(x)
    pred = getOutput(x, target_shape)
    m = Model(inputs=input_array, outputs=pred, name=name)
    return m
