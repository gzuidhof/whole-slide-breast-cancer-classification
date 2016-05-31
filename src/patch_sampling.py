import time
#from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
from tqdm import tqdm
import numpy as np

import multiresolutionimageinterface as mir

from deepr.data_processing.wsi_data_sources import WholeSlideImageDataSource, WholeSlideImageClassSampler, WholeSlideImageRandomPatchExtractor
from deepr.data_processing.simple_operations import LambdaVoxelOperation
from deepr.data_processing.batch_generator import RandomBatchGenerator
from deepr.data_processing.batch_adapter import BatchAdapterLasagne
from deepr.data_processing.simple_operations import OrdinalLabelVectorizer

import util
import dataset

nr_classes=3
labels_dict = {q:q+1 for q in range(nr_classes)}

def process(tra_fl, msk_src, network_parameters):
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
    print "Done in ", time.time()-s

    patch_extractor = WholeSlideImageRandomPatchExtractor(tra_wsi, tra_msk)



    train_data_source = LambdaVoxelOperation(patch_extractor, name = "image normalizer",
                                 input_names = ["image"],
                                 label_names = [],
                                 function = util.normalize_named_image)

    final_data_source = OrdinalLabelVectorizer(train_data_source, "label", "label", nr_classes)
    batch_generator = RandomBatchGenerator([final_data_source]) 
     
    batch_generator_lasagne = BatchAdapterLasagne(batch_generator)
    batch_generator_lasagne.select_inputs(["image"])
    batch_generator_lasagne.select_labels(["label"])

    batch_generator_lasagne.get_batch(9)

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

def unet_generator(wsi_filenames, msk_src, patch_size, crop=None):
    
    filenames = [(im,msk_src[im]) for im in wsi_filenames]
    
    def genny(filenames):
        
        return extract_random_patches(filenames, patch_size, crop)
    
    return genny

def extract_random_patch(filename_tuple, patch_size, crop_size=None):
    image_filename, mask_filename = filename_tuple
    
    r = mir.MultiResolutionImageReader()
    level=0
    img = r.open(image_filename)
    dims = img.getLevelDimensions(level)
    
    max_x = np.max((patch_size,dims[0]-patch_size))
    place = np.random.randint(0,max_x)

    image = img.getUCharPatch(place, place, patch_size, patch_size, level)
    image = image.transpose(2,0,1) #From 0,1,c to c,0,1
    img.close()
    
    r = mir.MultiResolutionImageReader()
    level=0
    img = r.open(mask_filename)
    #dims = img.getLevelDimensions(level)
    mask = img.getUCharPatch(place, place, patch_size, patch_size, level)
    #print dims
    mask = mask.transpose(2,0,1)
    img.close()
    
    if crop_size is not None:
        offset = (mask.shape[1]-crop_size)//2
        mask = mask[:,offset:offset+crop_size,offset:offset+crop_size]
    image = np.expand_dims(image, axis=0)
    image = np.array(image, dtype=np.float32)
    image = util.normalize_image(image)
    image = util.zero_center(image)
    image = util.random_flips(image)
    
    background_mask = np.where(mask==-1,0,1)
    mask = np.clip(np.array(np.expand_dims(mask, axis=0), dtype=np.int64)-1,0,100)
    
    
    weights = np.ones_like(mask, dtype=np.float32)
    weights = np.array(weights*background_mask,dtype=np.float32)
    return image, mask, weights
    
    
def extract_random_patches(filenames, patch_size, crop_size=None):
    data_points = [extract_random_patch(f, patch_size, crop_size) for f in filenames]
    images, masks, weights = zip(*data_points)
  
    return np.concatenate(images,axis=0), np.concatenate(masks,axis=0), np.concatenate(weights,axis=0)

def get_filenames(network_parameters):
    Benign_file_list, DCIS_file_list, IDC_file_list = dataset.train_filenames(shuffle=True)
    Benign_val_file_list, DCIS_val_file_list, IDC_val_file_list = dataset.validation_filenames(shuffle=True)
    
    msk_fls_All = dataset.mask_folder()
    msk_src = {}

    n_val_samples = network_parameters.num_val_samples
    n_train_samples = network_parameters.num_train_samples

    random_evaluation_items, msk_src = dataset.per_class_filelist(Benign_val_file_list, DCIS_val_file_list, IDC_val_file_list, msk_fls_All, msk_src, n_val_samples)
    random_train_items, msk_src = dataset.per_class_filelist(Benign_file_list, DCIS_file_list, IDC_file_list, msk_fls_All, msk_src, n_train_samples)
    
    filenames_val = [(im,msk_src[im]) for im in random_evaluation_items]
    filenames_train = [(im,msk_src[im]) for im in random_train_items]
    
    return filenames_train, filenames_val


def prepare_sampler(network_parameters):
    Benign_file_list, DCIS_file_list, IDC_file_list = dataset.train_filenames(shuffle=True)
    Benign_val_file_list, DCIS_val_file_list, IDC_val_file_list = dataset.validation_filenames(shuffle=True)

    msk_fls_All = dataset.mask_folder()
    msk_src = {}

    n_val_samples = network_parameters.num_val_samples
    n_train_samples = network_parameters.num_train_samples
    
    random_evaluation_items, msk_src = dataset.per_class_filelist(Benign_val_file_list, DCIS_val_file_list, IDC_val_file_list, msk_fls_All, msk_src, n_val_samples)
    random_train_items, msk_src = dataset.per_class_filelist(Benign_file_list, DCIS_file_list, IDC_file_list, msk_fls_All, msk_src, n_train_samples)
    
    print "Loading validation masks"
    batch_generator_lasagne_validation = prepare_lasagne_patch(random_evaluation_items, msk_src, network_parameters, multiprocess=True, processes=4)
    validation_generator = partial(gen, batch_generator_lasagne=batch_generator_lasagne_validation)
    
    print "Loading train masks"
    batch_generator_lasagne_train = prepare_lasagne_patch(random_train_items, msk_src, network_parameters, multiprocess=True, processes=4)
    train_generator = partial(gen, batch_generator_lasagne=batch_generator_lasagne_train)
        
    return train_generator, validation_generator
