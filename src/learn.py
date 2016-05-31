from __future__ import division
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import scipy.misc
from functools import partial
import metrics
from params import Params
import patch_sampling

if __name__ == "__main__":
    import theano
    import theano.tensor as T
    import lasagne
    import unet
    from unet import INPUT_SIZE, OUTPUT_SIZE

from tqdm import tqdm
from glob import glob

import cPickle as pickle
from parallel import ParallelBatchIterator

def calc_dice(train,truth):
    score = np.sum(train[truth>0])*2.0 / (np.sum(train) + np.sum(truth))
    return score

def make_dir_if_not_present(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    # create Theano variables for input and target minibatch
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets', dtype='int64')
    weight_var = T.tensor4('weights')

    print "Defining network"
    net_dict = unet.define_network(input_var)
    network = net_dict['out']

    train_fn, val_fn = unet.define_updates(network, input_var, target_var, weight_var)

    model_name = 'unet'+str(int(time.time()))
    model_folder = os.path.join('../data/models',model_name)
    plot_folder = os.path.join('../images/plot',model_name)

    folders = ['../images','../images/plot','../data','../data/models', model_folder, plot_folder]
    map(make_dir_if_not_present, folders)

    np.random.seed(0)
    
    experiment_file = '../exp/Experiment5-UNet.txt'
    print "Loading parameters from", experiment_file
    network_parameters = Params(experiment_file)

    filenames_train, filenames_val = patch_sampling.get_filenames(network_parameters)

    train_batch_size = 9
    val_batch_size = 18

    num_epochs = 400

    metric_names = ['Loss  ','L2    ','Accuracy','Dice  ','Precision','Recall']
    train_metrics_all = []
    val_metrics_all = []
    
    generator = partial(patch_sampling.extract_random_patches, patch_size=INPUT_SIZE, crop_size=OUTPUT_SIZE)
    print("Starting training...")
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_batches = 0
        start_time = time.time()

        np.random.shuffle(filenames_train)
        train_gen = ParallelBatchIterator(generator, filenames_train, ordered=False, batch_size=train_batch_size, multiprocess=True, n_producers=6)

        train_metrics = []
        val_metrics = []

        for i, batch in enumerate(tqdm(train_gen)):
            inputs, targets, weights = batch
            err, l2_loss, acc, dice, true, prob, prob_b = train_fn(inputs, targets, weights)
            tp,tn,fp,fn = metrics.calc_errors(true, prob_b)

            train_metrics.append([err, l2_loss, acc, dice, tp, tn, fp, fn])
            train_batches += 1

            if np.ceil(i/train_batch_size) % 10 == 0:
                im = np.hstack((
                    true[:OUTPUT_SIZE**2].reshape(OUTPUT_SIZE,OUTPUT_SIZE),
                    prob[:OUTPUT_SIZE**2][:,1].reshape(OUTPUT_SIZE,OUTPUT_SIZE)))

                plt.imsave(os.path.join(plot_folder,'train_{}_epoch{}.png'.format(model_name, epoch)),im)

        # And a full pass over the validation data:
        val_batches = 0

        np.random.shuffle(filenames_val)
        val_gen = ParallelBatchIterator(generator, filenames_val, ordered=True, batch_size=val_batch_size,multiprocess=False)

        for i, batch in enumerate(tqdm(val_gen)):
            inputs, targets, weights = batch

            err, l2_loss, acc, dice, true, prob, prob_b = val_fn(inputs, targets, weights)
            tp,tn,fp,fn = metrics.calc_errors(true, prob_b)

            val_metrics.append([err, l2_loss, acc, dice, tp, tn, fp, fn])
            val_batches += 1
            
            if np.ceil(i/val_batch_size) % 10 == 0: #Create image every 10th image
                im = np.hstack((
                    true[:OUTPUT_SIZE**2].reshape(OUTPUT_SIZE,OUTPUT_SIZE),
                    prob[:OUTPUT_SIZE**2][:,1].reshape(OUTPUT_SIZE,OUTPUT_SIZE)))

                plt.imsave(os.path.join(plot_folder,'val_{}_epoch{}.png'.format(model_name, epoch)),im)
                plt.close()

        train_metrics = np.sum(np.array(train_metrics),axis=0)/train_batches
        val_metrics = np.sum(np.array(val_metrics),axis=0)/val_batches

        precision_train = train_metrics[4] / (train_metrics[4]+train_metrics[6])
        recall_train = train_metrics[4] / (train_metrics[4]+train_metrics[7])

        precision_val = val_metrics[4] / (val_metrics[4]+val_metrics[6])
        recall_val = val_metrics[4] / (val_metrics[4]+val_metrics[7])

        train_metrics = list(train_metrics[:4]) + [precision_train,recall_train] #Strip off false positives et al
        val_metrics = list(val_metrics[:4]) + [precision_val,recall_val]

        # Then we print the results for this epoch:
        print("\nEpoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

        #print "Metrics"
        for name, train_metric, val_metric in zip(metric_names, train_metrics, val_metrics):
            print " {}:\t\t {:.6f}\t{:.6f}".format(name,train_metric,val_metric)

        if epoch % 4 == 0:
            print "Saving model"
            np.savez(os.path.join(model_folder,'{}_epoch{}.npz'.format(model_name, epoch)), *lasagne.layers.get_all_param_values(network))

        train_metrics_all.append(train_metrics)
        val_metrics_all.append(val_metrics)

        for name, train_vals, val_vals in zip(metric_names, zip(*train_metrics_all),zip(*val_metrics_all)):
            plt.figure()
            plt.plot(train_vals)
            plt.plot(val_vals)
            plt.ylabel(name)
            plt.xlabel("Epoch")

            plt.savefig(os.path.join(plot_folder, '{}.png'.format(name)))
            plt.close()

