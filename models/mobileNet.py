from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.activations import relu, tanh
from models.components.mobileNet import dsConv2D, invertedResidualBlock
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, LeakyReLU
from models.components.common import getInput, flattenAll, getOutput, concatenate_together, expandForCNN


def getModel(cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim,
             ocean_dim=None, other_dim=None, target_shape=1, name="resnet"):
    # input to the model
    input_array = getInput(cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim,
                           ocean_dim)
    if other_dim:
        other_in = Input(shape=other_dim, name="OtherParam")
        input_array.append(other_in)

    x = concatenate_together([expandForCNN(i) for i in input_array])
    for _ in range(8):
        x = invertedResidualBlock(x, in_channel=1, out_channel=8, kernel_size=3, expansion_rate=8, strides=(1, 1))

    x = dsConv2D(x, in_channel=8, out_channel=8, kernel_size=3, strides=1, padding="valid")
    x = dsConv2D(x, in_channel=8, out_channel=16, kernel_size=3, strides=1, padding="valid")
    x = dsConv2D(x, in_channel=16, out_channel=16, kernel_size=3, strides=1, padding="valid")
    x = dsConv2D(x, in_channel=16, out_channel=32, kernel_size=3, strides=1, padding="valid")
    x = dsConv2D(x, in_channel=32, out_channel=32, kernel_size=3, strides=1, padding="valid")
    # x = dsConv2D(x, in_channel=32, out_channel=64, kernel_size=3, strides=1, padding="valid")
    # x = dsConv2D(x, in_channel=64, out_channel=64, kernel_size=3, strides=1, padding="valid")

    x = Flatten()(x)
    # unify output layer
    x = Dense(256, activation=relu)(x)

    # last stage processing
    x = LeakyReLU()(Dense(128)(x))
    pred = getOutput(x, target_shape)
    m = Model(inputs=input_array, outputs=pred, name=name)
    return m
