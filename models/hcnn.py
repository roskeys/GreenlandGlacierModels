from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.activations import tanh, relu
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from models.components.common import AutoSetDenseOrCNN, getInput, flattenAll, getOutput, concatenate_together


def getModel(cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim,
             ocean_dim=None, other_dim=None, target_shape=1, name="hcnn"):
    # input to the model
    input_array = getInput(cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim, ocean_dim)
    if other_dim:
        other_in = Input(shape=other_dim, name="OtherParam")
        input_array.append(other_in)

    # CNN
    x = [AutoSetDenseOrCNN(i, horizontal=True, dropout=False, activation=relu, padding="valid") for i in input_array]
    x1 = concatenate_together(list(filter(lambda i: len(i.shape) == 4, x)), axis=1)
    x2 = concatenate_together(list(filter(lambda i: len(i.shape) != 4, x)), axis=1)
    x3 = Conv2D(16, kernel_size=(2, 1), padding='same', activation=relu)(x1)
    x4 = Dense(x2.shape[1] * 2)(x2)
    x = flattenAll([x3, x4])

    # last stage processing
    x = Dense(64, activation=tanh)(x)
    x = Dropout(0.5)(x)
    pred = getOutput(x, target_shape)
    m = Model(inputs=input_array, outputs=pred, name=name)
    return m


if __name__ == '__main__':
    model = getModel([12], [12], [12], [41, 12, 1], [41, 12, 1], [41, 12, 1],
                     ocean_dim=[8, 12, 1], other_dim=3)
    model.compile(optimizer='rmsprop', loss="mse")
