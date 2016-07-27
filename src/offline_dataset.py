import numpy as np
import h5py
from params import params as P
import patch_sampling
from parallel import ParallelBatchIterator
from tqdm import tqdm
import util
np.random.seed(0)
P.DATA_LEVEL=0

BATCH_SIZE = 300


f = h5py.File("breast_dataset_train.h5", 'w', libver='latest')
dset = f.create_dataset("patches_augmented", (3000000, 3, 224, 224), chunks=(300,3,224,224), maxshape=(None, 3, 224, 224), compression="gzip")

train_generator, validation_generator = patch_sampling.prepare_sampler()


X_train = [BATCH_SIZE]*10000
train_gen = ParallelBatchIterator(train_generator, X_train, ordered=False,
                                                batch_size=1,
                                                multiprocess=False,
                                                n_producers=4)

for i, batch in enumerate(tqdm(train_gen)):
    images, labels = batch
    images = util.unzero_center(images, P.MEAN_PIXEL)


    dset[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = np.array(images*255, dtype=np.int8)