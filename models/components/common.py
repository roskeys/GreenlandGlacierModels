from tensorflow import expand_dims
from tensorflow.keras import Input
from tensorflow.keras.activations import relu, tanh
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Concatenate


def getInput(cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim,
             ocean_dim=None):
    cloud_in = Input(shape=cloud_dim, name="Cloud")
    precipitation_in = Input(shape=precipitation_dim, name="Precipitation")
    wind_in = Input(shape=wind_dim, name="Wind")
    humidity_in = Input(shape=humidity_dim, name="Humidity")
    pressure_in = Input(shape=pressure_dim, name="Pressure")
    temperature_in = Input(shape=temperature_dim, name="Temperature")
    input_array = [cloud_in, precipitation_in, wind_in, humidity_in, pressure_in, temperature_in]
    if ocean_dim:
        ocean_in = Input(shape=ocean_dim, name="Ocean")
        input_array.append(ocean_in)
    return input_array


def AutoSetDenseOrCNN(x, horizontal=True, dropout=None, activation=relu, padding="valid", kernel_shape=None):
    shape = x.shape
    if len(shape) == 4:
        if kernel_shape is None:
            kernel_shape = (1, shape[2]) if horizontal else (shape[1], 1)
        out = Conv2D(16, kernel_size=kernel_shape, padding=padding, activation=activation)(x)
    else:
        flatten = Flatten()(x)
        out = Dense(shape[1] * 2)(flatten)
        if Dropout:
            out = Dropout(dropout)(out)
    return out


def flattenAll(x_list):
    return Concatenate(axis=1)([Flatten()(i) for i in x_list])


def concatenate_together(x_list, axis=1):
    shape = x_list[0].shape
    for i in x_list[1:]:
        if len(i.shape) != len(shape):
            return flattenAll(x_list)
    if len(shape) == 4:
        return Concatenate(axis=axis)(x_list)
    return flattenAll(x_list)


def getOutput(x, target_shape=1):
    x = Dense(32, activation=tanh)(x)
    x = Dropout(0.5)(x)
    # output prediction
    pred = Dense(target_shape)(x)
    return pred


def expandForCNN(x):
    if len(x.shape) == 4:
        return x
    assert 12 in x.shape
    if x.shape[1] == 12:
        x = expand_dims(expand_dims(x, 1), -1)
    assert len(x.shape) == 4
    return x
