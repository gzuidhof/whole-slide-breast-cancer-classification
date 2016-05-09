from __future__ import division
import lasagne
from lasagne.layers import InputLayer, DenseLayer, batch_norm

def define_network(featuremap_size_all, filter_size_all, dropout_ratio, image_size, input_var=None):
    #input 224

    network = lasagne.layers.InputLayer(shape=(None, 3, None, None),
                                        input_var=input_var)
                                        
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=featuremap_size_all[0], filter_size=(filter_size_all[0], filter_size_all[0]),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))  # known as Xavier initialization         
      
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=featuremap_size_all[1], filter_size=(filter_size_all[1], filter_size_all[1]),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))  # known as Xavier initialization     
      
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))    
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=featuremap_size_all[2], filter_size=(filter_size_all[2], filter_size_all[2]),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))           
    #network = lasagne.layers.dropout(network, p=0.3)            
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=featuremap_size_all[3], filter_size=(filter_size_all[3], filter_size_all[3]),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))           
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))    
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=featuremap_size_all[4], filter_size=(filter_size_all[4], filter_size_all[4]),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))           
    #network = lasagne.layers.dropout(network, p=0.3)            
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=featuremap_size_all[5], filter_size=(filter_size_all[5], filter_size_all[5]),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))               
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))    
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=featuremap_size_all[6], filter_size=(filter_size_all[6], filter_size_all[6]),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))           
    #network = lasagne.layers.dropout(network, p=0.3)            
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=featuremap_size_all[7], filter_size=(filter_size_all[7], filter_size_all[7]),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))         
    network = batch_norm(lasagne.layers.Conv2DLayer(
            network, num_filters=featuremap_size_all[8], filter_size=(filter_size_all[8], filter_size_all[8]),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))
                 
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))          
         
    network = batch_norm(lasagne.layers.Conv2DLayer(
            lasagne.layers.dropout(network, p = dropout_ratio),
            num_filters=filter_size_all[9], filter_size=(9, 9),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))
                 
    network = batch_norm(lasagne.layers.Conv2DLayer(
            lasagne.layers.dropout(network, p = dropout_ratio),
            num_filters=filter_size_all[10], filter_size=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal('relu')))   
                 
    network = lasagne.layers.Conv2DLayer(
            network,
            num_filters=filter_size_all[11], filter_size=(1, 1),
            nonlinearity=lasagne.nonlinearities.identity,
            W=lasagne.init.HeNormal())
                 
    return network  
    
def init_network(network, newmodel):
    old = lasagne.layers.get_all_param_values(network)
    new = []
    for layer in old:
      shape = layer.shape
      if len(shape)<2:
        shape = (shape[0], 1)
      W = lasagne.init.GlorotUniform()(shape)
      if W.shape != layer.shape:
        W = np.squeeze(W, axis= 1)
      new.append(W)
    lasagne.layers.set_all_param_values(network, new)
    return network    