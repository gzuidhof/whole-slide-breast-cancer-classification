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
import patch_sampling
import logging
from logger import initialize_logger
from params import params as P

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
import util

if __name__ == "__main__":
    np.random.seed(0)
    model_name = P.MODEL_ID

    model_folder = os.path.join('../models',model_name)
    plot_folder = os.path.join(model_folder, 'plots')
    image_folder = os.path.join(model_folder, 'images')

    folders = ['../models', model_folder, plot_folder, image_folder]
    map(util.make_dir_if_not_present, folders)
    initialize_logger(os.path.join(model_folder, 'log.txt').format(model_name))

    P.write_to_file(os.path.join(model_folder, 'config.ini'))
    logging.info(P.to_string())
    
    # create Theano variables for input and target minibatch
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets', dtype='int64')
    weight_var = T.tensor4('weights')

    logging.info("Defining network")
    net_dict = unet.define_network(input_var)
    network = net_dict['out']

    train_fn, val_fn = unet.define_updates(network, input_var, target_var, weight_var)

    filenames_train, filenames_val = patch_sampling.get_filenames()

    train_batch_size = P.BATCH_SIZE_TRAIN
    val_batch_size = P.BATCH_SIZE_VALIDATION

    num_epochs = 400

    metric_names_initial = ['Loss','L2','Accuracy','Dice']
    train_metrics = metrics.Metrics("train", metric_names_initial, n_classes=3)
    val_metrics = metrics.Metrics("validation", metric_names_initial, n_classes=3)
    
    generator = partial(patch_sampling.extract_random_patches, patch_size=P.INPUT_SIZE, crop_size=OUTPUT_SIZE)
    
    
    logging.info("Starting training...")
    for epoch in range(P.N_EPOCHS):
        # In each epoch, we do a full pass over the training data:
        start_time = time.time()

        np.random.shuffle(filenames_train)
        train_gen = ParallelBatchIterator(generator, filenames_train, ordered=False,
                                            batch_size=P.BATCH_SIZE_TRAIN,
                                            multiprocess=P.MULTIPROCESS_LOAD_AUGMENTATION,
                                            n_producers=P.N_WORKERS_LOAD_AUGMENTATION)


        for i, batch in enumerate(tqdm(train_gen)):
            inputs, targets, weights, _ = batch

            err, l2_loss, acc, dice, true, prob, prob_b = train_fn(inputs, targets, weights)

            train_metrics.append([err, l2_loss, acc, dice])
            train_metrics.append_prediction(true, prob_b)

            if i % 10 == 0:
                im = np.hstack((
                    true[:OUTPUT_SIZE**2].reshape(OUTPUT_SIZE,OUTPUT_SIZE),
                    prob[:OUTPUT_SIZE**2][:,1].reshape(OUTPUT_SIZE,OUTPUT_SIZE)))

                plt.imsave(os.path.join(image_folder,'train_epoch{}.png'.format(epoch)),im)

        metric_names, train_values = train_metrics.batch_done()

        # And a full pass over the validation data:
        np.random.shuffle(filenames_val)
        val_gen = ParallelBatchIterator(generator, filenames_val, ordered=False,
                                            batch_size=P.BATCH_SIZE_VALIDATION,
                                            multiprocess=P.MULTIPROCESS_LOAD_AUGMENTATION,
                                            n_producers=P.N_WORKERS_LOAD_AUGMENTATION)

        for i, batch in enumerate(tqdm(val_gen)):
            inputs, targets, weights, _ = batch

            err, l2_loss, acc, dice, true, prob, prob_b = val_fn(inputs, targets, weights)

            val_metrics.append([err, l2_loss, acc, dice])
            val_metrics.append_prediction(true, prob_b)

            if i % 30 == 0: #Create image every 10th image
                im = np.hstack((
                    true[:OUTPUT_SIZE**2].reshape(OUTPUT_SIZE,OUTPUT_SIZE),
                    prob[:OUTPUT_SIZE**2][:,1].reshape(OUTPUT_SIZE,OUTPUT_SIZE)))

                plt.imsave(os.path.join(image_folder,'val_epoch{}.png'.format(epoch)),im)
                plt.close()

        metric_names, val_values = val_metrics.batch_done()

        # Then we print the results for this epoch:
        logging.info("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, P.N_EPOCHS, time.time() - start_time))

        for name, train_metric, val_metric in zip(metric_names, train_values, val_values):
            name = name.rjust(20," ") #Pad the name until 10 characters long
            logging.info("{}:\t {:.6f}\t{:.6f}".format(name,train_metric,val_metric))

        if epoch % P.SAVE_EVERY_N_EPOCH == 0:
            logging.info("Saving model")
            np.savez_compressed(os.path.join(model_folder,'{}_epoch{}.npz'.format(model_name, epoch)), *lasagne.layers.get_all_param_values(network))

        labels, train_values_all = train_metrics.values_per_epoch()
        labels, val_values_all = val_metrics.values_per_epoch()

        for label, train_vals, val_vals in zip(labels, train_values_all,val_values_all):
            plt.figure()
            plt.plot(train_vals)
            plt.plot(val_vals)
            plt.ylabel(label)
            plt.xlabel("Epoch")

            plt.savefig(os.path.join(plot_folder, '{}.png'.format(label)))
            plt.close()

        logging.info("\n")