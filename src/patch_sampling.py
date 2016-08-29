import time
#from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
from tqdm import tqdm
import numpy as np

import util
import dataset
#import loss_weighting
import augment

from params import params as P
import gzip
import cPickle as pickle
import sys
import os
sys.path.append('../src/sampler')

import time
from wsi_random_patch_sampler import WSIRandomPatchSampler

def custom_generator(batch_size, wsi_random_patch_sampler, deterministic=False):
    images, labels, filenames = wsi_random_patch_sampler.sample_n_balanced(batch_size)
    images = util.normalize_image(images)

    if P.AUGMENT and not deterministic:
        images = augment.augment(images)

    if P.ZERO_CENTER:
        util.zero_center(images, P.MEAN_PIXEL)
    
    # 0-index the labels
    labels -= 1

    return images.astype(np.float32), labels.astype(np.int32), filenames

def prepare_custom_sampler(mini_subset=False, override_cache_size=None):
    """
        Returns two functions, one for train and one for validation. Call it with an integer as
        argument (the amount of images you want (probably one batch)).
    """
    s = time.time()
    
    file_unf = os.path.join(P.SAMPLER_FOLDER, 'samplers_{0}_{1}x{2}_nclass_{3}.pkl.gz')
    train_filename = file_unf.format('train' if not mini_subset else 'train_mini', P.PIXELS, P.PIXELS, P.N_CLASSES)
    val_filename = file_unf.format('validation' if not mini_subset else 'validation_mini', P.PIXELS, P.PIXELS, P.N_CLASSES)

    print "Loading train samplers"
    with gzip.open(train_filename,'rb') as f:
        samplers_train = pickle.load(f)

    print "Loading validation samplers"
    with gzip.open(val_filename,'rb') as f:
        samplers_val = pickle.load(f)

    print "Loading samplers took {} seconds.".format(time.time() - s)

    labels = [1,2,3][:P.N_CLASSES]

    train_sampler = WSIRandomPatchSampler(samplers_train, labels=labels, override_cache_size=override_cache_size)
    val_sampler = WSIRandomPatchSampler(samplers_val, labels=labels, override_cache_size=override_cache_size)

    train_generator = partial(custom_generator, wsi_random_patch_sampler=train_sampler)
    val_generator = partial(custom_generator, wsi_random_patch_sampler=val_sampler, deterministic=True)

    return train_generator, val_generator
        