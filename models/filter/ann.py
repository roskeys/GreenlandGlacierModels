from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu
from models.components.common import getInput, flattenAll


def getANNFilter(cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim, ocean_dim=None,
                 other_dim=None, target_shape=256, name="ann", horizontal=True):
    # input to the model
    input_array = getInput(cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim,
                           ocean_dim)
    if other_dim:
        other_in = Input(shape=other_dim, name="OtherParam")
        input_array.append(other_in)

    # flatten all data and concatenate together
    x = flattenAll(input_array)
    x = Dense(512, activation=relu)(x)

    # unify output layer
    out = Dense(target_shape, activation=relu)(x)
    return Model(inputs=input_array, outputs=out, name=name)
