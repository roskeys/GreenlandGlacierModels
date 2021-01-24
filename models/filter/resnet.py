from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.activations import relu
from models.components.ResNet import ResidualBlock
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from models.components.common import getInput, flattenAll, concatenate_together, expandForCNN


def getModel(cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim,
             ocean_dim=None, other_dim=None, target_shape=256, name="resnet", horizontal=True):
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

    # unify output layer
    out = Dense(target_shape, activation=relu)(x)
    return Model(inputs=input_array, outputs=out, name=name)
