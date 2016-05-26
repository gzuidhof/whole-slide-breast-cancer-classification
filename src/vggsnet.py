from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
import lasagne.layers

from mirror_padding import MirrorPadLayer


def define_network(network_parameters, input_var):
    net = {}

    net = InputLayer((None, 3, network_parameters.image_size, network_parameters.image_size), input_var=input_var)
    
    #net = MirrorPadLayer(net, width=(4,4))
    
    net = ConvLayer(net,
                             num_filters=32,
                             filter_size=5,
                             stride=1,
                             pad=0,
                             flip_filters=False)
    #net = PoolLayer(net,
    #                         pool_size=2,
    #                         stride=2,
    #                         ignore_border=False)
    
    #net = MirrorPadLayer(net, width=(2,2))
    net = ConvLayer(net,
                             num_filters=64,
                             filter_size=3,
                             pad=0,
                             flip_filters=False)
                 
    net= PoolLayer(net,
                             pool_size=2,
                             stride=2,
                             ignore_border=False)
                 
    net = MirrorPadLayer(net, width=(2,2))
    net = ConvLayer(net,
                             num_filters=128,
                             filter_size=3,
                             pad=0,
                             flip_filters=False)
    
                             
                             

                             
    print lasagne.layers.get_output_shape(net)
                             
    net = DenseLayer(net, num_units=512)
    #net = DropoutLayer(net, p=0.5)
    net = DenseLayer(net, num_units=3, nonlinearity=None)
    #net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net
