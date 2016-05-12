import time

from deepr.data_processing.wsi_data_sources import WholeSlideImageDataSource, WholeSlideImageClassSampler, WholeSlideImageRandomPatchExtractor
from deepr.data_processing.simple_operations import LambdaVoxelOperation
from deepr.data_processing.batch_generator import RandomBatchGenerator
from deepr.data_processing.batch_adapter import BatchAdapterLasagne
from deepr.data_processing.simple_operations import OrdinalLabelVectorizer

import util
#from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
from tqdm import tqdm

nr_classes=3
#labels_dict = {0:1, 1:2, 2:3}
labels_dict = {q:q+1 for q in range(nr_classes)}

def process(tra_fl, msk_src, network_parameters):
#def process(tra_fl):
    wsi = WholeSlideImageDataSource(tra_fl, (network_parameters.image_size, network_parameters.image_size), network_parameters.data_level)
    msk = WholeSlideImageClassSampler(msk_src[tra_fl], 0, nr_classes, labels_dict)
    return wsi, msk

def prepare_lasagne_patch(random_train_items, msk_src, network_parameters, multiprocess=True):

    print "getting all masks"
    s = time.time()
    if multiprocess:
        pool = Pool(processes=4)
        process_partial = partial(process, msk_src=msk_src, network_parameters=network_parameters)
        result = pool.map(process_partial, random_train_items)
        tra_wsi, tra_msk = zip(*result)
        pool.close()
    else:
        tra_wsi = []
        tra_msk = []
        for tra_fl in tqdm(random_train_items): # 20X
            tra_wsi.append(WholeSlideImageDataSource(tra_fl, (network_parameters.image_size, network_parameters.image_size), network_parameters.data_level))
            tra_msk.append(WholeSlideImageClassSampler(msk_src[tra_fl], 0, nr_classes, labels_dict))
    print "Done in ", time.time()-s

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

    #if multiprocess:
        #pool.close()
    print "... is done"
    return batch_generator_lasagne
