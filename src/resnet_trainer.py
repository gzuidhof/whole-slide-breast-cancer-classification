from __future__ import division
import time
import numpy as np
import trainer
from params import params as P
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
import patch_sampling
import logging
import scipy.misc
from parallel import ParallelBatchIterator
from tqdm import tqdm
import os.path

if __name__ == "__main__":
    import theano
    import theano.tensor as T
    import lasagne
    import resnet
    import patch_sampling
    from resnet import LR_SCHEDULE

class ResNetTrainer(trainer.Trainer):
    def __init__(self):
        metric_names = ['Loss','L2','Accuracy']
        super(ResNetTrainer, self).__init__(metric_names)

        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

        logging.info("Defining network")
        net = resnet.ResNet_FullPre_Wide(input_var, P.DEPTH, P.BRANCHING_FACTOR)
        self.network = net
        train_fn, val_fn, l_r = resnet.define_updates(self.network, input_var, target_var)

        self.train_fn = train_fn
        self.val_fn = val_fn
        self.l_r = l_r

    def do_batches(self, fn, batch_generator, metrics):
        for i, batch in enumerate(tqdm(batch_generator)):
            inputs, targets = batch
            targets = np.array(np.argmax(targets, axis=1), dtype=np.int32)
            err, l2_loss, acc, prediction, _ = fn(inputs, targets)

            metrics.append([err, l2_loss, acc])
            metrics.append_prediction(targets, prediction)

    def train(self, generator_train, X_train, generator_val, X_val):
        #filenames_train, filenames_val = patch_sampling.get_filenames()
        #generator = partial(patch_sampling.extract_random_patches, patch_size=P.INPUT_SIZE, crop_size=OUTPUT_SIZE)

        logging.info("Starting training...")
        for epoch in range(P.N_EPOCHS):
            self.pre_epoch()

            if epoch in LR_SCHEDULE:
                logging.info("Setting learning rate to {}".format(LR_SCHEDULE[epoch]))
                self.l_r.set_value(LR_SCHEDULE[epoch])
            #Full pass over the training data
            train_gen = ParallelBatchIterator(generator_train, X_train, ordered=False,
                                                batch_size=1,
                                                multiprocess=P.MULTIPROCESS_LOAD_AUGMENTATION,
                                                n_producers=P.N_WORKERS_LOAD_AUGMENTATION)

            self.do_batches(self.train_fn, train_gen, self.train_metrics)

            # And a full pass over the validation data:
            val_gen = ParallelBatchIterator(generator_val, X_val, ordered=False,
                                                batch_size=1,
                                                multiprocess=P.MULTIPROCESS_LOAD_AUGMENTATION,
                                                n_producers=P.N_WORKERS_LOAD_AUGMENTATION)

            self.do_batches(self.val_fn, val_gen, self.val_metrics)
            self.post_epoch()



if __name__ == "__main__":
    train_generator, validation_generator = patch_sampling.prepare_sampler()

    X_train = [P.BATCH_SIZE_TRAIN]*(P.EPOCH_SAMPLES_TRAIN//P.BATCH_SIZE_TRAIN)
    X_val = [P.BATCH_SIZE_VALIDATION]*(P.EPOCH_SAMPLES_VALIDATION//P.BATCH_SIZE_VALIDATION)

    trainer = ResNetTrainer()
    trainer.train(train_generator, X_train, validation_generator, X_val)