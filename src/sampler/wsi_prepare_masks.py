import sys
sys.path.append('../')

import dataset
import util
import wsi_mask
import wsi_sampler

from multiprocessing.pool import ThreadPool as Pool
import cPickle as pickle


PATCH_SIZE = (224,224)
BORDER_DISTANCE = PATCH_SIZE[0]//2
DATA_LEVEL = 1

# Amount of samples loaded at once per file open
CACHE_SIZE = 4

if False:
    for out_file, files in zip(
        ['/mnt/diskB/guido/samplers_train.pkl.gz', '/mnt/diskB/guido/samplers_validation.pkl.gz'], 
        [dataset.train_filenames(), dataset.validation_filenames()]):


        print len(files[0]), len(files[1]), len(files[2]) #Sanity check
        labels = [1]*len(files[0]) + [2]*len(files[1]) + [3]*len(files[2])

        # Nones are for no subset, we want to prepare masks for all data
        filenames, mask_sources = dataset.per_class_filelist(files[0],files[1],files[2],dataset.mask_folder(),{},(None,None,None))

        

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
        #print len(labels)
        #print len(filenames)
        pool = Pool(6)
        samplers = list(util.pool_progress(pool, create_sampler, zip(filenames,labels)))
        samplers = filter(None, samplers)
        pool.close()
        pool.join()

        print "Now saving to file (will take a while).."

        import gzip
        with gzip.open(out_file,'wb') as f:
            pickle.dump(samplers,f, protocol=2)

for out_file, files in zip(
    ['/mnt/diskB/guido/samplers_train_mini.pkl.gz', '/mnt/diskB/guido/samplers_validation_mini.pkl.gz'], 
    [dataset.train_filenames(), dataset.validation_filenames()]):


    print len(files[0]), len(files[1]), len(files[2]) #Sanity check
    labels = [1]*30 + [2]*30 + [3]*30

    filenames, mask_sources = dataset.per_class_filelist(files[0],files[1],files[2],dataset.mask_folder(),{},(30,30,30))

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
