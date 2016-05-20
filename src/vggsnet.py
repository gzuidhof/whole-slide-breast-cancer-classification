from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax


def define_network(input_var):
    net = {}

    net['input'] = InputLayer((None, 3, 224, 224), input_var=input_var)
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=64,
                             filter_size=5,
                             stride=2,
                             flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1'],
                             pool_size=2,
                             stride=2,
                             ignore_border=False)
    net['conv3'] = ConvLayer(net['pool1'],
                             num_filters=128,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    net['conv5'] = ConvLayer(net['conv3'],
                             num_filters=256,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5'],
                             pool_size=3,
                             stride=3,
                             ignore_border=False)
    net['fc7'] = DenseLayer(net['pool5'], num_units=1024)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=3, nonlinearity=None)
    #net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net
