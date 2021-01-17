from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.activations import tanh, relu
from models.components.common import getInput, flattenAll, getOutput


def getModel(cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim,
             ocean_dim=None, other_dim=None, target_shape=1, name="ann"):
    # input to the model
    input_array = getInput(cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim, ocean_dim)
    if other_dim:
        other_in = Input(shape=other_dim, name="OtherParam")
        input_array.append(other_in)
    # flatten all data and concatenate together
    x = flattenAll(input_array)
    x = Dense(x.shape[1], activation=tanh)(x)
    x = Dropout(0.5)(x)

    # last stage processing
    x = Dense(64, activation=relu)(x)
    x = Dropout(0.5)(x)
    pred = getOutput(x, target_shape)
    m = Model(inputs=input_array, outputs=pred, name=name)
    return m


if __name__ == '__main__':
    model = getModel([12], [12], [12], [41, 12, 1], [41, 12, 1], [41, 12, 1],
                     ocean_dim=[8, 12, 1])
    model.compile(optimizer='rmsprop', loss="mse")
    import numpy as np

    x = [np.random.rand(128, 12), np.random.rand(128, 12), np.random.rand(128, 12),
         np.random.rand(128, 41, 12, 1), np.random.rand(128, 41, 12, 1), np.random.rand(128, 41, 12, 1),
         np.random.rand(128, 8, 12, 1)]
    y = np.random.rand(128, 1)

    x_t = [np.random.rand(16, 12), np.random.rand(16, 12), np.random.rand(16, 12),
           np.random.rand(16, 41, 12, 1), np.random.rand(16, 41, 12, 1), np.random.rand(16, 41, 12, 1),
           np.random.rand(16, 8, 12, 1)]
    y_t = np.random.rand(16, 1)
    model.fit(x, y, validation_data=(x_t, y_t), epochs=5)

