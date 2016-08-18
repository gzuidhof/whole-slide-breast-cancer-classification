import sys
sys.path.append('../')
from params import params as P

import dataset
import util
import wsi_mask
import wsi_sampler

from multiprocessing.pool import ThreadPool as Pool
import cPickle as pickle
import os

PATCH_SIZE = (P.INPUT_SIZE, P.INPUT_SIZE) #(224,224)
BORDER_DISTANCE = PATCH_SIZE[0]//2
DATA_LEVEL = 0

# Amount of samples loaded at once per file open
# Increases speed, but also memory usage
CACHE_SIZE = 4

out_file_unformatted = os.path.join(P.SAMPLER_FOLDER, 'samplers_{0}_{1}x{2}_nclass_{3}.pkl.gz')

train_filenames = dataset.train_filenames()
validation_filenames = dataset.validation_filenames()

if P.N_CLASSES == 2:
    label_numbers = [1,2,2]
else:
     label_numbers = [1,2,3]

print "-"*20
print "N CLASSES:", P.N_CLASSES
print "-"*20

#########
# Prepare sampler for full dataset
#########

if True:
    for name in ['train', 'train_mini', 'validation', 'validation_mini']:

        if 'train' in name:
            files = dataset.train_filenames()

        if 'validation' in name:
            files = validation_filenames = dataset.validation_filenames()


        print "\n\nNow preparing sampler {}...".format(name)
        out_file = out_file_unformatted.format(name, PATCH_SIZE[0], PATCH_SIZE[1], P.N_CLASSES)
        print "Out file: {}".format(out_file)

        if 'mini' in name:
            subsets = (30,30,30)
            labels = [label_numbers[0]]*30 + [label_numbers[1]]*30 + [label_numbers[2]]*30
        else:
            subsets = (None,None,None)
            labels = [label_numbers[0]]*len(files[0]) + [label_numbers[1]]*len(files[1]) + [label_numbers[2]]*len(files[2])

        print "Amount per class (total):", len(files[0]), len(files[1]), len(files[2]) #Sanity check
        print "Subsets:", subsets #Sanity check

        # Nones are for no subset, we want to prepare masks for all data
        filenames, mask_sources = dataset.per_class_filelist(files[0],files[1],files[2],dataset.mask_folder(),{}, subsets)

        def create_sampler(filename_label_tuple):
            wsi_filename, label = filename_label_tuple
            mask_filename = mask_sources[wsi_filename]
            try:
                mask = wsi_mask.WSIMask(mask_filename, border_distance=BORDER_DISTANCE, data_level=DATA_LEVEL, labels=[label])
                sampler = wsi_sampler.WSISampler(wsi_filename, mask, DATA_LEVEL, PATCH_SIZE, cache_size=CACHE_SIZE)
                return sampler
            except IndexError:
                print "Dropping file {}, probably not large enough for this data level".format(wsi_filename)
                return None

        pool = Pool(6)
        samplers = list(util.pool_progress(pool, create_sampler, zip(filenames,labels))) 
        samplers = filter(None, samplers)
        pool.close()
        pool.join()

        print "Now saving to file (will take a while).."

        import gzip
        with gzip.open(out_file,'wb') as f:
            pickle.dump(samplers,f, protocol=2)