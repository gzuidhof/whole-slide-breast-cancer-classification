# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 16:02:52 2015

@author: Babak & Guido
"""

if __name__ == "__main__":
    import theano
    import theano.tensor as T
    import lasagne
    from lasagne.layers import InputLayer, DenseLayer, batch_norm
    from lasagne.nonlinearities import softmax #leaky_rectify,

    import joblib

    import numpy as np
    import os
    import time
    #import progressbar
    #from time import sleep
    import ntpath
    import random
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score

    #from deeplearning.training_extensions.boosting import OnlineBooster
    #from PIL import Image
    if os.name == 'nt':
        sys.path.append("C:\\Program Files\\ASAP 1.4.0\\bin")
        sys.path.append("../..")

    import multiresolutionimageinterface as mir

    #import resource
    #resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

    #sys.path.append(r"C:\Users\francesco\Data")
    sys.setrecursionlimit(1000000) #Set the maximum depth of the Python interpreter stack to limit. This limit prevents infinite recursion from causing an overflow of the C stack and crashing Python.
    np.random.seed(0)

    import ntpath

    import util
    import dataset
    import babysitting
    import net
    import patch_sampling
    from parallel import ParallelBatchIterator




def train(num_batches_tra, batch_generator_lasagne, batch_size):
    train_accu = 0
    train_err = 0
    train_l2 = 0
    train_batches = 0
    start_time = time.time()
    augmentation_time = 0
    
    batch_gen = ParallelBatchIterator(batch_generator_lasagne, X=[batch_size]*num_batches_tra)
    for i, inputs in enumerate(batch_gen):
    #for i in range(num_batches_tra):
        s = time.time()
        #inputs = batch_generator_lasagne.get_batch(batch_size)
        inputs[0].values()[0] = util.random_flips(inputs[0].values()[0])

        util.zero_center(inputs[0].values()[0])
        augmentation_time+=time.time()-s

        err_loss, l2_loss, predictions, acc = train_fn(inputs[0].values()[0].astype("float32"), inputs[1].values()[0].astype("float32"))
        train_err += err_loss
        train_accu += acc
        train_l2 += l2_loss
        train_batches += 1
        percent = (i+1) / float(num_batches_tra)
        babysitting.progress_bar(percent, 50, err_loss, l2_loss, acc, 1)


        #print "\n",x / (time.time()-s)


    # Then we print the results for this epoch:
    print("  Mini epoch took {:.3f}s".format(time.time() - start_time))
    print("  Loading and augmentation took {:.3f}s".format(augmentation_time))
    print("  training accuracy: {:.6f} ---- training loss: {:.6f} ---- training L2 loss: {:.6f}".format(train_accu / train_batches, train_err / train_batches, train_l2 / train_batches))
    #print("  training L2 loss :\t\t{:.6f}".format(train_l2 / train_batches))
    train_loss = train_err / train_batches
    train_accuracy = train_accu / train_batches
    return train_loss, train_accuracy


if __name__ == "__main__":
    from params import Params
    if os.name == 'posix':
        path = r'/media/sf_Guido/exp/test.txt'
    else:
        path = r'../exp/Experiment31-Patch224.txt'
    experiment = path.rsplit('/', 1)[1][:-4]
    #import glob
    #print glob.glob('../exp/*')
    #print os.getcwd()
    network_parameters = Params(path)

    learning_rate = theano.shared(np.float32(network_parameters.learning_rate_schedule_adam[0]))
    dropout_ratio = theano.shared(np.float32(network_parameters.dropout_ratio))
    l2_Lambda = theano.shared(np.float32(network_parameters.l2_Lambda))



    input_var= T.ftensor4('inputs')
    target_var = T.fmatrix('targets')
    #stored_param_values = joblib.load(r"F:\Code\JournalDCIS_IDC\Results\ToyNetwork\Experiment5-Patch224_best_network_single_scale.pkl_01.npy.z")
    network = net.define_network(network_parameters.featuremap_size_all, network_parameters.filter_size_all, dropout_ratio, network_parameters.image_size, input_var)

    #lasagne.layers.set_all_param_values(network, stored_param_values['param values'])

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
    val_fn = theano.function([input_var, target_var], [test_loss, test_out, test_acc])



###############################################################################################################
    # Get training image files from different categories

    from glob import glob

    if os.name == 'posix':
        saving_DIR = r'/media/sf_Guido/exp/test.txt'
    else:
        saving_DIR = '../exp/results/'

    #saving_dir = path

    Benign_file_list, DCIS_file_list, IDC_file_list = dataset.train_filenames(shuffle=True)
    Benign_val_file_list, DCIS_val_file_list, IDC_val_file_list = dataset.validation_filenames(shuffle=True)

    msk_fls_All = dataset.mask_folder()

    train_loss_lst = []
    train_accuracy_lst = []
    val_loss_lst = []
    val_acc_lst = []
    val_auc_lst = []
    epoch_lst = []
    epoch_val_lst = []
    best_val_acc = 0
    best_val_auc = 0
    best_train_acc = 0

    sys.stdout.flush()
    best_epoch = 0
    milestone_epoch = 0
    #lr_decay = (network_parameters.learning_rate_schedule_adam[1]/network_parameters.learning_rate_schedule_adam[0])**(1./network_parameters.n_epochs)
    lr_decay = network_parameters.lr_decay
    msk_src = {}

    val_num = 0
    start_time = time.time()
    print "Getting all validation data"
    random_evaluation_items, msk_src = dataset.per_class_filelist(Benign_val_file_list, DCIS_val_file_list, IDC_val_file_list, msk_fls_All, msk_src, network_parameters.num_val_samples, val_num)
    batch_generator_lasagne_Val = patch_sampling.prepare_lasagne_patch(random_evaluation_items, msk_src, network_parameters)
    print "All validation data in memory"
    print("Getting validation images took {:.3f}s".format(time.time() - start_time))

    mini_epoch = 0
    start_time = time.time()
    random_train_items, msk_src = dataset.per_class_filelist(Benign_file_list, DCIS_file_list, IDC_file_list, msk_fls_All, msk_src, network_parameters.num_train_samples, mini_epoch)
    batch_generator_lasagne_train = patch_sampling.prepare_lasagne_patch(random_train_items, msk_src, network_parameters)
    print("Getting training images took {:.3f}s".format(time.time() - start_time))

    for epoch_num in range(network_parameters.n_epochs):
        print '\n------ Epoch '+str(epoch_num+1)+' ------'

        trainingFolds = 1
        if epoch_num==0:
            num_of_train_iterator = network_parameters.num_of_train_iterator[0]
            num_of_val_iterator = network_parameters.num_of_val_iterator[0]
        else:
            num_of_train_iterator = network_parameters.num_of_train_iterator[1]
            num_of_val_iterator = network_parameters.num_of_val_iterator[1]

        print 'Learning rate set to: ' +str(learning_rate.get_value())
        for mini_epoch in range(trainingFolds):
            print '------ Training on Epoch {:d} ------'.format(epoch_num+1)
            train_loss, train_accuracy = train(num_of_train_iterator, batch_generator_lasagne_train, network_parameters.batch_size)
            if epoch_num == 0:
                train_loss = 1.0
            train_loss_lst.append(train_loss)
            train_accuracy_lst.append(train_accuracy)
            epoch_lst.append(epoch_num*trainingFolds + (mini_epoch+1))

#            train_loss_plot, = plt.plot(epoch_lst, train_loss_lst,'r')
#            train_accuracy_plot, = plt.plot(epoch_lst, train_accuracy_lst,'#FFA500')
#            filename = experiment + '_training_plot.png'
#            plt.savefig(os.path.join(saving_DIR, filename))


            #if mini_epoch==1 or mini_epoch==2:
            val_loss = []
            val_acc = []
            ValidationFolds = 1
            for val_num in range(ValidationFolds):
                print '************ Validating Epoch {:d} of ***********'.format(epoch_num+1), experiment
                # And a full pass over the validation data:
                val_err = 0
                acc = 0
                val_batches = 0#0
                all_predictions = []
                all_truth = []
                for k in range(num_of_val_iterator):
                    inputs = batch_generator_lasagne_Val.get_batch(network_parameters.batch_size*3)
                    util.zero_center(inputs[0].values()[0])


                    err, test_prediction, t_acc = val_fn(inputs[0].values()[0].astype("float32"), inputs[1].values()[0].astype("float32"))
                    all_predictions.append(test_prediction)
                    all_truth.append(inputs[1].values()[0].astype("float32"))

                    val_err += err
                    acc += t_acc
                    val_batches += 1
                    percent = (k+1) / float(num_of_val_iterator)
                    babysitting.progress_bar(percent, 50, err, t_acc)

                unlisted_predictions = [item[0] for sublist in all_predictions for item in sublist]
                unlisted_all_truth = [item[0] for sublist in all_truth for item in sublist]
                val_auc = roc_auc_score(unlisted_all_truth, unlisted_predictions)
                val_loss.append(val_err / val_batches)
                val_acc.append(acc / val_batches)

            val_loss_lst.append(sum(val_loss) / len(val_loss))
            val_acc_lst.append(sum(val_acc) / len(val_acc))
            val_auc_lst.append(val_auc)
            epoch_val_lst.append(epoch_num * trainingFolds + (mini_epoch+1))

            print '\n'
            print 'Overall validation AUC on epoch {:d} is {:.4f}'.format(epoch_num+1, val_auc)
            print 'Best validation AUC was obtained in epoch {:d} with auc {:.4f}'.format(best_epoch, best_val_auc)
            print 'Overall validation accuracy on epoch {:d} is {:.4f}'.format(epoch_num+1, val_acc_lst[-1])
            print 'Overall training accuracy on epoch {:d} was '.format(epoch_num+1) + str(train_accuracy)

            filename = experiment + '_Latest_network_single_scale.pkl'
            joblib.dump(network, os.path.join(saving_DIR, filename),2)
            if val_auc_lst[-1] > best_val_auc:
                print 'validation AUC improved: saving network'
                filename = experiment + '_best_network_single_scale.pkl'
                joblib.dump(network, os.path.join(saving_DIR, filename),2)
                best_val_acc = val_acc_lst[-1]
                best_val_auc = val_auc_lst[-1]
                best_epoch = epoch_num + 1

            if ((train_accuracy_lst[-1] - best_train_acc) >= network_parameters.accuracy_tolerance):
                best_train_acc = train_accuracy_lst[-1]
                milestone_epoch = epoch_num + 1


            #log = '/mnt/rdstorage1/Userdata/Babak/BreastDataset/Results/' + experiment + '_log.txt'
            #with open(log, "a") as f:
            #    f.write('Overall validation accuracy on epoch {:d}-{:d} was {:.4f}\n'.format(epoch_num+1, mini_epoch+1, val_acc_lst[-1]))
            #    f.write('Best validation AUC was obtained in epoch {:d} with AUC {:.4f}\n'.format(best_epoch, best_val_auc))
            #    f.write('Best validation accuracy was obtained in epoch {:d} with accuracy {:.4f}\n'.format(best_epoch, best_val_acc))
            #    f.write("============================================================================================================\n")

        #plt.legend()



        fig = plt.figure()
        plt.title('IDC-DCIS system, Best train acc = {:.4f}, Best Val AUC {:.4f}'.format(train_accuracy, best_val_auc))
        line1, = plt.plot(0,2,'r', label="train_loss")
        line2, = plt.plot(0,0,'#FFA500', label="train_accuracy")
        line3, = plt.plot(0,0,'g', label="val_accuracy")
        line4, = plt.plot(0,0,'m', label="val_AUC")
        plt.legend(handles=[line1, line2, line3, line4], loc=4)

        train_loss_plot, = plt.plot(epoch_lst,train_loss_lst,'r')
        train_accuracy_plot, = plt.plot(epoch_lst, train_accuracy_lst,'#FFA500')
        val_acc_plot, = plt.plot(epoch_val_lst,val_acc_lst,'g')
        val_auc_plot, = plt.plot(epoch_val_lst,val_auc_lst,'m')

        filename = experiment + '_training_plot.png'
        plt.savefig(os.path.join(saving_DIR, filename))
        plt.close("all")
        if (epoch_num - milestone_epoch) >= network_parameters.epoch_tolerance:
            milestone_epoch = epoch_num
            learning_rate.set_value(np.float32(learning_rate.get_value()*network_parameters.lr_decay))
            dropout_ratio.set_value(np.float32(dropout_ratio.get_value()*network_parameters.DropOut_decay))
            l2_Lambda.set_value(np.float32(l2_Lambda.get_value()*network_parameters.L2_decay))
