from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.activations import tanh
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
    x = Dense(512, activation=tanh)(x)

    x = Dense(256, activation=tanh)(x)
    x = Dropout(0.2)(x)
    # last stage processing
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    pred = getOutput(x, target_shape)
    m = Model(inputs=input_array, outputs=pred, name=name)
    return m
