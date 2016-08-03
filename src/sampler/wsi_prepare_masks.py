import sys
sys.path.append('../')

import dataset
import util
import wsi_mask
import wsi_sampler

from multiprocessing.pool import ThreadPool as Pool
import cPickle as pickle



for out_file, files in zip(
    ['/mnt/diskB/guido/samplers_train.pkl.gz', '/mnt/diskB/guido/samplers_validation.pkl.gz'], 
    [dataset.train_filenames(), dataset.validation_filenames]):



    print len(files[0]), len(files[1]), len(files[2]) #Sanity check

    # Nones are for no subset, we want to prepare masks for all data
    files, mask_sources = dataset.per_class_filelist(files[0],files[1],files[2],dataset.mask_folder(),{},(None,None,None))

    def create_sampler(wsi_filename):
        mask_filename = mask_sources[wsi_filename]
        mask = wsi_mask.WSIMask(mask_filename, border_distance=112)
        sampler = wsi_sampler.WSISampler(wsi_filename, mask, 0, (224,244))
        return sampler

    pool = Pool(6)
    samplers = list(util.pool_progress(pool, create_sampler, files))
    pool.close()
    pool.join()

    import gzip
    with gzip.open(out_file,'wb') as f:
        pickle.dump(samplers,f, protocol=2)