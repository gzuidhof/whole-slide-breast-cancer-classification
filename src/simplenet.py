import sys
sys.setrecursionlimit(10000)
import lasagne
from lasagne.nonlinearities import rectify, softmax, sigmoid
from lasagne.layers import InputLayer, MaxPool2DLayer, DenseLayer, DropoutLayer, helper, batch_norm, BatchNormLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer, ElemwiseSumLayer, NonlinearityLayer, PadLayer, GlobalPoolLayer, ExpressionLayer
from lasagne.init import Orthogonal, HeNormal, GlorotNormal
from lasagne.updates import nesterov_momentum, adam

import theano
import theano.tensor as T
import numpy as np
from params import params as P

#LR_SCHEDULE = {
#    0: 0.0012,
#    6: 0.012,
#    80: 0.0012,
#    120: 0.00012,
#}
LR_SCHEDULE = {
    0: 0.001,
    8: 0.001,
    80: 0.0001,
    120: 0.00001,
}



PIXELS = P.INPUT_SIZE
imageSize = PIXELS * PIXELS
num_classes = P.N_CLASSES
n_channels = P.CHANNELS

he_norm = HeNormal(gain='relu')


def SimpleNet(input_var=None):
    # Building the network
    l = InputLayer(shape=(None, n_channels, PIXELS, PIXELS), input_var=input_var)

    #l = batch_norm(ConvLayer(l, num_filters=16, filter_size=(7,7), stride=(2,2), nonlinearity=rectify, pad='same', W=he_norm))
    #l = batch_norm(ConvLayer(l, num_filters=16, filter_size=(5,5), stride=(2,2), nonlinearity=rectify, pad='same', W=he_norm))

    l = MaxPool2DLayer(l, (4,4))

    #l = batch_norm(ConvLayer(l, num_filters=32, filter_size=(3,3), stride=(2,2), nonlinearity=rectify, pad='same', W=he_norm))
    #l = batch_norm(ConvLayer(l, num_filters=32, filter_size=(3,3), stride=(2,2), nonlinearity=rectify, pad='same', W=he_norm))

    #l = MaxPool2DLayer(l, (2,2))

    #l = batch_norm(ConvLayer(l, num_filters=64, filter_size=(3,3), stride=(2,2), nonlinearity=rectify, pad='same', W=he_norm))
    #l = batch_norm(ConvLayer(l, num_filters=64, filter_size=(3,3), stride=(2,2), nonlinearity=rectify, pad='same', W=he_norm))

    l = DenseLayer(l, num_units=3, W=HeNormal(), nonlinearity=rectify)
    #l = DenseLayer(l, num_units=256, W=HeNormal(), nonlinearity=rectify)

    network = DenseLayer(l, num_units=num_classes, W=HeNormal(), nonlinearity=softmax)

    return network

def define_updates(output_layer, X, Y):
    output_train = lasagne.layers.get_output(output_layer)
    output_test = lasagne.layers.get_output(output_layer, deterministic=True)

    #L2 regularization
    all_layers = lasagne.layers.get_all_layers(output_layer)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * P.L2_LAMBDA

    # set up the loss that we aim to minimize when using cat cross entropy our Y should be ints not one-hot
    loss = lasagne.objectives.categorical_crossentropy(T.clip(output_train,0.00001,0.99999), Y)
    loss = loss.mean()
    loss = loss + l2_penalty

    # set up loss functions for validation dataset
    test_loss = lasagne.objectives.categorical_crossentropy(T.clip(output_test,0.00001,0.99999), Y)
    test_loss = test_loss.mean()
    test_loss = test_loss + l2_penalty

    acc = T.mean(T.eq(T.argmax(output_train, axis=1), Y), dtype=theano.config.floatX)
    test_acc = T.mean(T.eq(T.argmax(output_test, axis=1), Y), dtype=theano.config.floatX)

    # get parameters from network and set up sgd with nesterov momentum to update parameters, l_r is shared var so it can be changed
    l_r = theano.shared(np.array(LR_SCHEDULE[0], dtype=theano.config.floatX))
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    #updates = nesterov_momentum(loss, params, learning_rate=l_r, momentum=P.MOMENTUM)
    updates = adam(loss, params, learning_rate=l_r)

    prediction_binary = T.argmax(output_train, axis=1)
    test_prediction_binary = T.argmax(output_test, axis=1)

    # set up training and prediction functions
    train_fn = theano.function(inputs=[X,Y], outputs=[loss, l2_penalty, acc, prediction_binary, output_train[:,1]], updates=updates)
    valid_fn = theano.function(inputs=[X,Y], outputs=[test_loss, l2_penalty, test_acc, test_prediction_binary, output_test[:,1]])

    return train_fn, valid_fn, l_r