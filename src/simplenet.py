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

LR_SCHEDULE = {
    0: 0.005,
    6: 0.005,
    80: 0.0001,
    120: 0.00001,
}

PIXELS = P.INPUT_SIZE
imageSize = PIXELS * PIXELS
num_classes = P.N_CLASSES
n_channels = P.CHANNELS

he_norm = HeNormal(gain='relu')

def SimpleNet(input_var=None):

    net = {}
    net['input'] = InputLayer((None, P.N_CLASSES, P.INPUT_SIZE, P.INPUT_SIZE), input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = Pool2DLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = Pool2DLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = Pool2DLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = Pool2DLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = Pool2DLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=512)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=512)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=P.N_CLASSES, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net['prob']

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