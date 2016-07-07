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
#import loss_weighting
import augment

from params import params as P

nr_classes=3
labels_dict = {q:q+1 for q in range(nr_classes)}

def process(tra_fl, msk_src):
    wsi = WholeSlideImageDataSource(tra_fl, (P.PIXELS, P.PIXELS), P.DATA_LEVEL)
    msk = WholeSlideImageClassSampler(msk_src[tra_fl], 0, nr_classes, labels_dict)
    return wsi, msk

def prepare_lasagne_patch(random_train_items, msk_src, multiprocess=True, processes=4):
    s = time.time()

    if multiprocess:
        pool = Pool(processes=processes)
        try:
            process_partial = partial(process, msk_src=msk_src)
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
        for tra_fl in tqdm(random_train_items):
            wsi = WholeSlideImageDataSource(tra_fl, (P.PIXELS, P.PIXELS), P.DATA_LEVEL)
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

    

    print "... is done"
    return batch_generator_lasagne

#Generates a batch of given size by calling supplied generator
def gen(batch_size, batch_generator_lasagne):
    batch = batch_generator_lasagne.get_batch(batch_size)

    images = batch[0].values()[0]

    if P.AUGMENT:
        images = augment.augment(images)

    offset = (images.shape[2] - P.INPUT_SIZE) // 2
            
    if offset > 0:
        images = images[:,:,offset:-offset,offset:-offset]

    if P.ZERO_CENTER:
        util.zero_center(images, P.MEAN_PIXEL)

    images = images.astype("float32")
    labels = batch[1].values()[0].astype("int32")

    return images, labels

def unet_generator(wsi_filenames, msk_src, patch_size, crop=None):
    
    filenames = [(im,msk_src[im]) for im in wsi_filenames]
    
    # A generator function, which returns images,masks,weights,filenames lists
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
    
    mask = np.array(np.expand_dims(mask, axis=0), dtype=np.int64)-1
    
    return image, mask
    
    
def extract_random_patches(filenames, patch_size, crop_size=None):
    data_points = [extract_random_patch(f, patch_size, crop_size) for f in filenames]
    images, masks = zip(*data_points)
    
    ims = np.concatenate(images,axis=0)
    msks = np.concatenate(masks,axis=0)
    
    weights = np.array(loss_weighting.weight_by_class_balance(msks, classes=[0,1,2]),dtype=np.float32)
    msks = np.clip(msks,0,100)
  
    return ims, msks, weights, filenames

def get_filenames():
    Benign_file_list, DCIS_file_list, IDC_file_list = dataset.train_filenames(shuffle=True)
    Benign_val_file_list, DCIS_val_file_list, IDC_val_file_list = dataset.validation_filenames(shuffle=True)
    
    msk_fls_All = dataset.mask_folder()
    msk_src = {}

    n_val_samples = P.SUBSET_VALIDATION
    n_train_samples = P.SUBSET_TRAIN

    random_evaluation_items, msk_src = dataset.per_class_filelist(Benign_val_file_list, DCIS_val_file_list, IDC_val_file_list, msk_fls_All, msk_src, n_val_samples)
    random_train_items, msk_src = dataset.per_class_filelist(Benign_file_list, DCIS_file_list, IDC_file_list, msk_fls_All, msk_src, n_train_samples)
    
    filenames_val = [(im,msk_src[im]) for im in random_evaluation_items]
    filenames_train = [(im,msk_src[im]) for im in random_train_items]
    
    return filenames_train, filenames_val


def prepare_sampler():
    Benign_file_list, DCIS_file_list, IDC_file_list = dataset.train_filenames(shuffle=True)
    Benign_val_file_list, DCIS_val_file_list, IDC_val_file_list = dataset.validation_filenames(shuffle=True)

    msk_fls_All = dataset.mask_folder()
    msk_src = {}

    n_val_samples = P.SUBSET_VALIDATION
    n_train_samples = P.SUBSET_TRAIN
    
    random_evaluation_items, msk_src = dataset.per_class_filelist(Benign_val_file_list, DCIS_val_file_list, IDC_val_file_list, msk_fls_All, msk_src, n_val_samples)
    random_train_items, msk_src = dataset.per_class_filelist(Benign_file_list, DCIS_file_list, IDC_file_list, msk_fls_All, msk_src, n_train_samples)
    
    print "Loading validation masks"
    batch_generator_lasagne_validation = prepare_lasagne_patch(random_evaluation_items, msk_src, multiprocess=True, processes=6)
    validation_generator = partial(gen, batch_generator_lasagne=batch_generator_lasagne_validation)
    
    print "Loading train masks"
    batch_generator_lasagne_train = prepare_lasagne_patch(random_train_items, msk_src, multiprocess=True, processes=6)
    train_generator = partial(gen, batch_generator_lasagne=batch_generator_lasagne_train)
        
    return train_generator, validation_generator
