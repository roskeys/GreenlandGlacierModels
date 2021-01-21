from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.activations import relu
from models.components.ResNet import ResidualBlock
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
        x = ResidualBlock(x, filters=8, kernel_size=3, strides=(1, 1), padding='same', shortcut=True)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), padding='same', activation=relu)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x) if other_dim is None else flattenAll([x, other_in])

    # last stage processing
    x = Dense(64)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    pred = getOutput(x, target_shape)
    m = Model(inputs=input_array, outputs=pred, name=name)
    return m
