from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.activations import relu, tanh
from models.components.ResNet import ResidualBlock
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, LeakyReLU, BatchNormalization, add
from models.components.common import getInput, flattenAll, getOutput, concatenate_together, expandForCNN


def dsConv2D(model, in_channel, out_channel, kernel_size, strides, padding = "valid"):
    """
    Depthwise Convolution 2D block
    args:
    model: the prior layers
    in_channel: the number of input channels
    out_channel: the number of output channels
    kernel_size: the kernel_size for first hidden layer
    stride: the stride for first hidden layer
    padding: padding for first hidden layer
    return:
    returns the model with dsConv2D layers added
    """
    # Depthwise
    block = Conv2D(filters = in_channel,
                   kernel_size = kernel_size,
                   strides = strides,
                   padding = padding,
                   groups = in_channel,
                   use_bias = False)(model)
    block = BatchNormalization(axis = 3)(block)
    block = ReLU()(block)
    # Pointwise
    block = Conv2D(filters = out_channel,
                   kernel_size = 1,
                   strides = (1, 1),
                   padding = "valid",
                   use_bias = False)(block)
    block = BatchNormalization(axis = 3)(block)
    block = ReLU(block)
    return block

def invertedResidualBlock(model, in_channel, out_channel, kernel_size, expansion_rate = 8, strides = (1, 1)):
    """
    Inverted Residual Block from Mobile Net V2
    args:
    model: the prior layers
    in_channel: the number of input channels
    out_channel: the number of output channels
    kernel_size: the kernel size for the 2nd hidden layer
    expansion_rate: expansion rate of channels for 2nd hidden layer
    strides: the stride for second hidden layer
    return:
    returns the model with inverted Redisual Block added
    """
    # Expansion layer
    block = Conv2D(filters = in_channel * expansion_rate,
                   kernel_size = 1,
                   strides = (1, 1),
                   use_bias = False)(model)
    block = BatchNormalization(axis = 3)(block)
    block = ReLU()(block)
    # Depthwise Layer
    block = Conv2D(filters = in_channel * expansion_rate,
                   kernel_size = kernel_size,
                   strides = strides,
                   padding = "same",
                   groups = in_channel * expansion_rate,
                   use_bias = False)(block)
    block = BatchNormalization(axis = 3)(block)
    block = ReLU()(block)
    # Linear Bottleneck
    block = Conv2D(filters = out_channel,
                   kernel_size = 1,
                   strides = (1, 1),
                   use_bias = False
                   )(block)
    block = BatchNormalization(axis = 3)(block)
    # shortcut
    if in_channel == out_channel:
        block = add([model, block])
        return block
    else:
        shortcut = Conv2D(filters = out_channel,
                          kernel_size = 1,
                          strides = (1, 1),
                          padding = "valid",
                          use_bias = False)(model)
        shortcut = BatchNormalization(axis = 3)(shortcut)
        block = add([shortcut, block])
