import time

from deepr.data_processing.wsi_data_sources import WholeSlideImageDataSource, WholeSlideImageClassSampler, WholeSlideImageRandomPatchExtractor
#from wsi_data_sources import WholeSlideImageDataSource, WholeSlideImageClassSampler, WholeSlideImageRandomPatchExtractor
from deepr.data_processing.simple_operations import LambdaVoxelOperation
from deepr.data_processing.batch_generator import RandomBatchGenerator
from deepr.data_processing.batch_adapter import BatchAdapterLasagne
from deepr.data_processing.simple_operations import OrdinalLabelVectorizer

import util
#from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
from tqdm import tqdm
import dataset

nr_classes=3
#labels_dict = {0:1, 1:2, 2:3}
labels_dict = {q:q+1 for q in range(nr_classes)}

def process(tra_fl, msk_src, network_parameters):
#def process(tra_fl):
    wsi = WholeSlideImageDataSource(tra_fl, (network_parameters.image_size, network_parameters.image_size), network_parameters.data_level)
    msk = WholeSlideImageClassSampler(msk_src[tra_fl], 0, nr_classes, labels_dict)
    return wsi, msk

def prepare_lasagne_patch(random_train_items, msk_src, network_parameters, multiprocess=True, processes=4):

    print "getting all masks"
    s = time.time()
    c = 0

    if multiprocess:
        pool = Pool(processes=processes)
        try:
            process_partial = partial(process, msk_src=msk_src, network_parameters=network_parameters)
            result = pool.map(process_partial, random_train_items)
            tra_wsi, tra_msk = zip(*result)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print "Caught KeyboardInterrupt, terminating workers"
            pool.terminate()
            pool.join()
    else:
        tra_wsi = []
        tra_msk = []
        for tra_fl in tqdm(random_train_items): # 20X
            c+=1
            wsi = WholeSlideImageDataSource(tra_fl, (network_parameters.image_size, network_parameters.image_size), network_parameters.data_level)
            msk = WholeSlideImageClassSampler(msk_src[tra_fl], 0, nr_classes, labels_dict)
            tra_wsi.append(wsi)
            tra_msk.append(msk)
            #print wsi
            #print msk
            #print tra_fl
            #if c > 1:
            #    break
            #break
    print "Done in ", time.time()-s

    #print train_wsi
    #print tra_msk
    patch_extractor = WholeSlideImageRandomPatchExtractor(tra_wsi, tra_msk)



    train_data_source = LambdaVoxelOperation(patch_extractor, name = "image normalizer",
                                 input_names = ["image"],
                                 label_names = [],
                                 function = util.normalize_image)

    final_data_source = OrdinalLabelVectorizer(train_data_source, "label", "label", nr_classes)
    batch_generator = RandomBatchGenerator([final_data_source]) 
     
    batch_generator_lasagne = BatchAdapterLasagne(batch_generator)
    batch_generator_lasagne.select_inputs(["image"])
    batch_generator_lasagne.select_labels(["label"])

    batch_generator_lasagne.get_batch(9)
    #if multiprocess:
        #pool.close()
    print "... is done"
    return batch_generator_lasagne

#Generates a batch of given size by calling supplied generator
def gen(batch_size, batch_generator_lasagne):
    batch = batch_generator_lasagne.get_batch(batch_size)
    batch[0].values()[0] = util.random_flips(batch[0].values()[0])
    util.zero_center(batch[0].values()[0])

    images = batch[0].values()[0].astype("float32")
    labels = batch[1].values()[0].astype("float32")
    return images, labels



def prepare_sampler(network_parameters):
    Benign_file_list, DCIS_file_list, IDC_file_list = dataset.train_filenames(shuffle=True)
    Benign_val_file_list, DCIS_val_file_list, IDC_val_file_list = dataset.validation_filenames(shuffle=True)

    msk_fls_All = dataset.mask_folder()

    val_num = 0
    mini_epoch = 0
    msk_src = {}

    n_val_samples = network_parameters.num_val_samples
    n_train_samples = network_parameters.num_train_samples



    #Skip loading evaluation
    print "Loading validation masks"
    random_evaluation_items, msk_src = dataset.per_class_filelist(Benign_val_file_list, DCIS_val_file_list, IDC_val_file_list, msk_fls_All, msk_src, n_val_samples, val_num)
    batch_generator_lasagne_validation = prepare_lasagne_patch(random_evaluation_items, msk_src, network_parameters, multiprocess=True, processes=4)

    validation_generator = partial(gen, batch_generator_lasagne=batch_generator_lasagne_validation)

    print "Loading train masks"
    random_train_items, msk_src = dataset.per_class_filelist(Benign_file_list, DCIS_file_list, IDC_file_list, msk_fls_All, msk_src, n_train_samples, mini_epoch)
    batch_generator_lasagne_train = prepare_lasagne_patch(random_train_items, msk_src, network_parameters, multiprocess=True, processes=4)

    train_generator = partial(gen, batch_generator_lasagne=batch_generator_lasagne_train)

    return train_generator, validation_generator
