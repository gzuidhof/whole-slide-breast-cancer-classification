import numpy as np
import ntpath
import util
import dataset
import babysitting
import net
import patch_sampling
from parallel import ParallelBatchIterator

from functools import partial
from params import Params
import time

def gen(batch_size, batch_generator_lasagne):
    batch = batch_generator_lasagne.get_batch(batch_size)
    batch[0].values()[0] = util.random_flips(batch[0].values()[0])
    util.zero_center(batch[0].values()[0])

    images = batch[0].values()[0].astype("float32")
    labels = batch[1].values()[0].astype("float32")
    return images, labels


if __name__ == "__main__":
    Benign_file_list, DCIS_file_list, IDC_file_list = dataset.train_filenames(shuffle=True)
    Benign_val_file_list, DCIS_val_file_list, IDC_val_file_list = dataset.validation_filenames(shuffle=True)

    Benign_file_list = Benign_file_list[:10]
    DCIS_file_list = DCIS_file_list[:10]
    IDC_file_list = IDC_file_list[:10]
    Benign_val_file_list = Benign_val_file_list[:10]
    DCIS_val_file_list = DCIS_val_file_list[:10]
    IDC_val_file_list = IDC_val_file_list[:10]

    msk_fls_All = dataset.mask_folder()

    msk_src = {}
    path = r'../exp/Experiment32-Patch224.txt'
    network_parameters = Params(path)
    val_num = 0
    mini_epoch = 0

    print Benign_file_list
    n_val_samples = [int(0.1*value) for value in network_parameters.num_val_samples]
    n_train_samples = [int(0.01*value) for value in network_parameters.num_train_samples]

    print "N validation samples", n_val_samples
    print "N train samples", n_train_samples

    #Skip loading evaluation
    #random_evaluation_items, msk_src = dataset.per_class_filelist(Benign_val_file_list, DCIS_val_file_list, IDC_val_file_list, msk_fls_All, msk_src, n_val_samples, val_num)
    #batch_generator_lasagne_Val = patch_sampling.prepare_lasagne_patch(random_evaluation_items, msk_src, network_parameters, multiprocess=False)

    random_train_items, msk_src = dataset.per_class_filelist(Benign_file_list, DCIS_file_list, IDC_file_list, msk_fls_All, msk_src, n_train_samples, mini_epoch)
    print len(random_train_items), random_train_items
    batch_generator_lasagne_train = patch_sampling.prepare_lasagne_patch(random_train_items, msk_src, network_parameters, multiprocess=True, processes=6)


    #batch = batch_generator_lasagne_train.get_batch(9)
    #print batch

    batch_size=90
    num_batches_tra = 10
    generator = partial(gen, batch_generator_lasagne=batch_generator_lasagne_train)
    batch_gen = ParallelBatchIterator(generator, X=[batch_size]*num_batches_tra,multiprocess=True)


    s = time.time()
    for x in batch_gen:
        print "Time to retrieve batch", time.time()-s
        print len(x)
        time.sleep(1)
        s = time.time()
