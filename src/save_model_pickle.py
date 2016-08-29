import resnet
import pickle
import lasagne
import numpy as np
import theano.tensor as T


print "Defining network"

input_var = T.tensor4('inputs')
target_var = T.ivector('targets')


net = resnet.ResNet_FullPre_Wide(input_var, 4, 2)
all_layers = lasagne.layers.get_all_layers(net)
net = all_layers[-3]
net = resnet.ResNet_Stacked(net)

print "Loading model"
model_save_file = '../models/1472001110_stacked/1472001110_stacked_epoch50.npz'#'../models/1470945732_resnet/1470945732_resnet_epoch478.npz'
with np.load(model_save_file) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(net, param_values)

all_layers = lasagne.layers.get_all_layers(net)

for x in range(1,9):
    i = -x
    print i, lasagne.layers.get_output_shape(all_layers[i])



#print "Saving model"
#with open('../models/768_model.pkl','wb') as f:
    #pickle.dump(net, f, 2)