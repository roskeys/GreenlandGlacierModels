from tensorflow.keras.layers import Conv2D, BatchNormalization, add


# Defines the Residual Block, revised
def ResidualBlock(model, filters, kernel_size, strides=(1, 1), padding='same', name=None, shortcut=False):
    bn_name = (name + "_bn") if name != None else None
    conv_name = (name + "_conv") if name != None else None

    # Conv->ReLU->BN
    block = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(model)
    block = BatchNormalization(axis=3, name=bn_name)(block)

    block = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(block)
    block = BatchNormalization(axis=3, name=bn_name)(block)

    if shortcut:
        shortcutBlock = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu',
                               name=conv_name)(model)
        shortcutBlock = BatchNormalization(axis=3, name=bn_name)(shortcutBlock)
        block = add([block, shortcutBlock])
    else:
        block = add([model, block])
    return block
