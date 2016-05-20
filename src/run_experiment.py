import sys
import os.path
from glob import glob
import numpy as np
np.random.seed(0)
import time
import logging

from params import Params
import patch_sampling
from parallel import ParallelBatchIterator
from tqdm import tqdm

if __name__ == "__main__":
    import theano
    import theano.tensor as T
    import lasagne

def load_network(network_parameters, input_var):
    print "Defining network"
    if network_parameters.architecture == "alexnet":
        import alexnet
        network = alexnet.define_network()
    elif network_parameters.architecture == "vggsnet":
        import vggsnet
        network = vggsnet.define_network(input_var)['fc8']
    elif network_parameters.architecture == "resnet":
        import resnet
        network = resnet.define_network(network_parameters, input_var)
    else:
        print "Unknown architecture"
        exit()
    return network


def define_learn_loss(network, network_parameters, input_var, target_var):
    learning_rate = theano.shared(np.float32(network_parameters.learning_rate_schedule_adam[0]))
    l2_Lambda = theano.shared(np.float32(network_parameters.l2_Lambda))

    prediction = lasagne.layers.get_output(network)
    e_x = T.exp(prediction - prediction.max(axis=1, keepdims=True))
    out = (e_x / e_x.sum(axis=1, keepdims=True)).flatten(2)

    # output is clipped to avoid ln of 0
    loss = lasagne.objectives.categorical_crossentropy(T.clip(out, 0.0001, 0.9999), target_var)
    l2_loss = l2_Lambda * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    loss = loss.mean() + l2_loss
    train_acc = T.mean(T.eq(T.argmax(out, axis=1), T.argmax(target_var, axis=1)),
                      dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(network, trainable=True)



    updates = lasagne.updates.adam(loss, params, learning_rate = learning_rate,
                                   beta1 = network_parameters.beta1, beta2 = network_parameters.beta2, epsilon = network_parameters.epsilon)

   # updates = lasagne.updates.nesterov_momentum(
   #         loss, params, learning_rate=0.005, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_e_x = T.exp(test_prediction - test_prediction.max(axis=1, keepdims=True))
    test_out = (test_e_x / test_e_x.sum(axis=1, keepdims=True)).flatten(2)

    # print test_out


    test_loss = lasagne.objectives.categorical_crossentropy(T.clip(test_out, 0.0001, 0.9999),
                                                            target_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_out, axis=1), T.argmax(target_var, axis=1)),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], [loss, l2_loss, out, train_acc], updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, l2_loss, test_out, test_acc])

    return train_fn, val_fn

def run_epoch(fn, n_batches, batch_generator, batch_size, multiprocess=True):
    start_time = time.time()
    augmentation_time = 0

    accu = 0
    err = 0
    l2 = 0
    batch_gen = ParallelBatchIterator(batch_generator, X=[batch_size]*n_batches, n_producers=12, max_queue_size=60)
    
    
    if multiprocess:
        print 'WITH PARALLEL'
        s = time.time()
        for i, (images, labels) in enumerate(tqdm(batch_gen)):
        #for i in tqdm(range(batch_size)):
        # (images, labels) = batch_generator(batch_size)
            augmentation_time+=time.time()-s
            err_loss, l2_loss, predictions, acc = fn(images, labels)
            err += err_loss
            accu += acc
            l2 += l2_loss
            s = time.time()
    else:
        print 'WITHOUT PARALLEL'
        s = time.time()
        for i in tqdm(range(batch_size)):
            (images, labels) = batch_generator(batch_size)
            augmentation_time+=time.time()-s
            err_loss, l2_loss, predictions, acc = fn(images, labels)
            err += err_loss
            accu += acc
            l2 += l2_loss
            s = time.time()

    print("  Mini epoch took {:.3f}s (loading and augmentation {:.3f}s)".format(time.time() - start_time, augmentation_time))
    logging.info("  Mini epoch took {:.3f}s (loading and augmentation {:.3f}s)".format(time.time() - start_time, augmentation_time))
    #print("  Loading and augmentation took {:.3f}s".format(augmentation_time))
    print("  accuracy: {:.6f} ---- loss: {:.6f} ---- L2 loss: {:.6f}".format(accu / n_batches, err / n_batches, l2 / n_batches))

    loss = err / n_batches
    accuracy = accu / n_batches

    return loss, accuracy



if __name__ == "__main__":
    experiment = sys.argv[1]

    candidates = glob('../exp/'+experiment+'*')
    if os.path.isfile(experiment):
        experiment_file = experiment
    elif len(candidates)==1:
        experiment_file = candidates[0]
    else:
        print "Experiment not found in", candidates
        exit()
        
    time_string = str(time.time())
    logging.basicConfig(filename='../exp/logs/{0}-{1}.log'.format(time_string,experiment),level=logging.DEBUG, format='%(asctime)s %(message)s')

    print "Loading parameters from", experiment_file
    network_parameters = Params(experiment_file)
    
    logging.info("parameters {}".format(network_parameters.dictionary))

    input_var= T.ftensor4('inputs')
    target_var = T.fmatrix('targets')

    network = load_network(network_parameters, input_var)
    print "Defining learn/validation functions"
    train_fn, val_fn = define_learn_loss(network, network_parameters, input_var, target_var)


    logging.info("creating samplers and loading masks")
    train_generator, validation_generator = patch_sampling.prepare_sampler(network_parameters)
    
    
    logging.info("starting training")
    for epoch_num in range(network_parameters.n_epochs):
        s = time.time()
        do_parallel = epoch_num % 2 == 0
        
        
        logging.info("epoch {}".format(epoch_num))
        print '\n------ Epoch '+str(epoch_num+1)+' ------'

        num_of_train_iterator = network_parameters.num_of_train_iterator[int(epoch_num<1)]
        num_of_val_iterator = network_parameters.num_of_val_iterator[int(epoch_num<1)]

        print '------ Training on Epoch {:d} ------'.format(epoch_num+1)
        
        train_loss, train_accuracy = run_epoch(train_fn, num_of_train_iterator, train_generator, network_parameters.batch_size, do_parallel)
        

        print '************ Validating Epoch {:d} of ***********'.format(epoch_num+1), experiment
        val_loss, val_accuracy = run_epoch(val_fn, num_of_val_iterator, validation_generator, network_parameters.batch_size, do_parallel)
        
        logging.info("epoch time {}".format(time.time()-s))
        logging.info("train_loss {}, train_accuracy {}".format(train_loss, train_accuracy))
        logging.info("val_loss {}, val_accuracy {}".format(val_loss, val_accuracy))
