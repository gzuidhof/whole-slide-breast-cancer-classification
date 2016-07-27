from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import batch_norm
from lasagne.nonlinearities import softmax
import lasagne.layers
from lasagne.updates import nesterov_momentum, adam

from mirror_padding import MirrorPadLayer

import theano
import theano.tensor as T
import numpy as np
from params import params as P


LR_SCHEDULE = {
    0: 0.01,
    80: 0.001,
    120: 0.0001,
}

def define_network(input_var, dropout_ratio=0.5):
    network = lasagne.layers.InputLayer(shape=(None, 3, None, None),
                                        input_var=input_var)
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=24, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))  # known as Xavier initialization         
 # 222
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=24, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))  # known as Xavier initialization    
 # 220            
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
 # 110    
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=48, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))
 # 108            
    #network = lasagne.layers.dropout(network, p=0.3)            
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=48, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))
 # 106            
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
 # 53    
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=96, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))
 # 51            
    #network = lasagne.layers.dropout(network, p=0.3)            
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=96, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))     
 # 49            
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
 # 25    
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=192, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))
 # 23            
    #network = lasagne.layers.dropout(network, p=0.3)            
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=192, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))         
 # 21
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=192, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))
 # 19    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
 # 10            
    network = batch_norm(lasagne.layers.Conv2DLayer(
            lasagne.layers.dropout(network, p = P.DROPOUT),
            num_filters=2048, filter_size=(9, 9),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))
            
    network = batch_norm(lasagne.layers.Conv2DLayer(
            lasagne.layers.dropout(network, p = P.DROPOUT),
            num_filters=1024, filter_size=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))   
            
    network = lasagne.layers.Conv2DLayer(
            network,
            num_filters=3, filter_size=(1, 1),
            nonlinearity=lasagne.nonlinearities.identity,
            W=lasagne.init.HeNormal())
            
    return network  


def define_updates(output_layer, X, Y):
    output_train = lasagne.layers.get_output(output_layer)
    output_test = lasagne.layers.get_output(output_layer, deterministic=True)

    e_x = np.exp(output_train - output_train.max(axis=1, keepdims=True))
    output_train = (e_x / e_x.sum(axis=1, keepdims=True)).flatten(2)

    e_x = np.exp(output_test - output_test.max(axis=1, keepdims=True))
    output_test = (e_x / e_x.sum(axis=1, keepdims=True)).flatten(2)


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