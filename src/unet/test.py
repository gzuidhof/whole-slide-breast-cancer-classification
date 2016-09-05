import unet
import theano.tensor as T

input_var = T.tensor4('inputs')
target_var = T.tensor4('targets', dtype='int64')
weight_var = T.tensor4('weights')

net = unet.define_network(input_var)

t, v = unet.define_updates(net['out'], input_var, target_var, weight_var)