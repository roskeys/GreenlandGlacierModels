from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.activations import relu, tanh
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, LeakyReLU, MaxPooling2D
from models.components.common import AutoSetDenseOrCNN, getInput, flattenAll, getOutput, concatenate_together


def getModel(cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim,
             ocean_dim=None, other_dim=None, target_shape=1, name="vcnn"):
    # input to the model
    input_array = getInput(cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim,
                           ocean_dim)
    if other_dim:
        other_in = Input(shape=other_dim, name="OtherParam")
        input_array.append(other_in)

    # CNN
    x = [AutoSetDenseOrCNN(i, horizontal=False, dropout=False, activation=relu, padding="valid") for i in input_array]
    x1 = concatenate_together(list(filter(lambda i: len(i.shape) == 4, x)), axis=1)
    x2 = concatenate_together(list(filter(lambda i: len(i.shape) != 4, x)), axis=1)

    x1 = MaxPooling2D(pool_size=(1, 2))(
        Conv2D(16, kernel_size=(1, 2), padding='same', activation=relu)(x1)) if x1 is not None else None
    x2 = Dense(x2.shape[1] * 2, activation=relu)(x2) if x2 is not None else None
    if x1 is not None and x2 is not None:
        x = flattenAll([x1, x2])
    else:
        x = Flatten()(x1) if x1 is not None else Flatten()(x2)

    # unify output layer
    x = Dense(256, activation=relu)(x)

    # last stage processing
    x = LeakyReLU()(Dense(128)(x))
    pred = getOutput(x, target_shape)
    m = Model(inputs=input_array, outputs=pred, name=name)
    return m
