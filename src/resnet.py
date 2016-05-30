from __future__ import division
import lasagne
from lasagne.layers import InputLayer, DenseLayer, batch_norm

from lasagne.layers import Conv2DLayer as ConvLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm

from mirror_padding import MirrorPadLayer

[[[[1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]]]]

def lambda_one(X):
    #return X[:, :, ::2, ::2]
    return X[:, :, :, :]

def lambda_two(s):
    return (s[0], s[1], s[2], s[3])

def residual_block(l, increase_dim=False, projection=False, mirror_pad=False):
    input_num_filters = l.output_shape[1]
    
    # print "Stride=2", increase_dim
    if increase_dim:
        #first_stride = (2,2)
        first_stride = (1,1)
        out_num_filters = input_num_filters*2
    else:
        first_stride = (1,1)
        out_num_filters = input_num_filters


    
    stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, W=lasagne.init.HeNormal(gain='relu')))
    stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, W=lasagne.init.HeNormal(gain='relu')))

    # add shortcut connections
    if increase_dim:
        if projection:
            pass
            # projection shortcut, as option B in paper
            #projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, b=None))
            #block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection],cropping=(None,None,'center', 'center')), nonlinearity=rectify)
        else:
            # identity shortcut, as option A in paper
            #identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
            # print "Padding"
            identity = ExpressionLayer(l, lambda_one, lambda_two)
            print lasagne.layers.get_output_shape(identity)
            padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
            
            
            if mirror_pad:
                padded_stack_2 = MirrorPadLayer(stack_2, (4,4))
            else: 
                padded_stack_2 = stack_2
            
            print lasagne.layers.get_output_shape(padding)
            print lasagne.layers.get_output_shape(padded_stack_2)
            # print lasagne.layers.get_output_shape(padding)
            
            block = NonlinearityLayer(ElemwiseSumLayer([padded_stack_2, padding],cropping=(None,None,'center', 'center')), nonlinearity=rectify)
    else:
        # print lasagne.layers.get_output_shape(stack_2)
        #print lasagne.layers.get_output_shape(l)
        block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l],cropping=(None,None,'center', 'center')), nonlinearity=rectify)

    return block


def define_network(params, input_var=None, n=6):
    #input 224

    # Building the network
    l_in = InputLayer(shape=(None, 3, params.image_size, params.image_size), input_var=input_var)

    # first layer, output is 64
    l = batch_norm(ConvLayer(l_in, num_filters=64, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, W=lasagne.init.HeNormal(gain='relu')))
    
    #l = MaxPool2DLayer(l, pool_size=2, stride=2)

    # first stack of residual blocks, output is 64
    for _ in range(n):
        l = residual_block(l)
        
    

    # second stack of residual blocks, output is 128
    l = residual_block(l, increase_dim=True)
    
    
    for _ in range(n+1):
        l = residual_block(l)
        
    l = MaxPool2DLayer(l, pool_size=2, stride=2)

    # third stack of residual blocks, output is 256
    l = residual_block(l, increase_dim=True)
    
    for _ in range(n+5):
        l = residual_block(l)
        
    # fourth stack of residual blocks, output is 512
    l = residual_block(l, increase_dim=True, mirror_pad=True)
    
    for _ in range(n-1):
        l = residual_block(l, mirror_pad=True)

    # average pooling
    l = GlobalPoolLayer(l)

    # fully connected layer
    network = DenseLayer(
            l, num_units=params.n_classes,
            W=lasagne.init.HeNormal(),
            nonlinearity=softmax)

    return network
