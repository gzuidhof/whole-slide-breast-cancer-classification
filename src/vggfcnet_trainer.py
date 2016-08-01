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
from cparallel import ContinuousParallelBatchIterator
from tqdm import tqdm
import os.path

if __name__ == "__main__":
    import theano
    import theano.tensor as T
    import lasagne
    import vggfcnet
    import patch_sampling
    from vggfcnet import LR_SCHEDULE
    from dataset import label_name

class VGGFCNetTrainer(trainer.Trainer):
    def __init__(self):
        metric_names = ['Loss','L2','Accuracy']
        super(VGGFCNetTrainer, self).__init__(metric_names)

        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

        logging.info("Defining network")
        net = vggfcnet.define_network(input_var)
        self.network = net
        train_fn, val_fn, l_r = vggfcnet.define_updates(self.network, input_var, target_var)

        self.train_fn = train_fn
        self.val_fn = val_fn
        self.l_r = l_r

    def save_debug_images(self, n, images, labels, metrics):
        im = images.transpose(0,2,3,1)

        im[:,:,:,0] += P.MEAN_PIXEL[0]
        im[:,:,:,1] += P.MEAN_PIXEL[1]
        im[:,:,:,2] += P.MEAN_PIXEL[2]

        f, axarr = plt.subplots(4,4,figsize=(12,12))
        for i in range(16):
            x = i%4
            y = i//4
            axarr[y,x].imshow(im[i])
            axarr[y,x].set_title(label_name(labels[i]))
            axarr[y,x].axis('off')
        
        #print np.mean(im)
        plt.subplots_adjust(wspace = -0.3, hspace=0.15)

        plt.savefig(os.path.join(self.image_folder, '{}_{}.png'.format(metrics.name,self.epoch, n)))
        plt.close()
        

    def do_batches(self, fn, batch_generator, metrics):
        
        batch_size = P.EPOCH_SAMPLES_TRAIN//P.BATCH_SIZE_TRAIN if metrics.name=='train' else P.EPOCH_SAMPLES_VALIDATION//P.BATCH_SIZE_VALIDATION

        for i, batch in enumerate(tqdm(batch_generator(batch_size))):
            inputs, targets = batch

            #print np.sum(targets,axis=0)
            targets = np.array(np.argmax(targets, axis=1), dtype=np.int32) #non one-hot
            #print targets
            err, l2_loss, acc, prediction, _ = fn(inputs, targets)

            metrics.append([err, l2_loss, acc])
            metrics.append_prediction(targets, prediction)

            if i == 0:
                self.save_debug_images(i, inputs, targets, metrics)

    def train(self, generator_train, X_train, generator_val, X_val):

        train_gen = ContinuousParallelBatchIterator(generator_train, ordered=False,
                                                batch_size=1,
                                                multiprocess=P.MULTIPROCESS_LOAD_AUGMENTATION,
                                                n_producers=P.N_WORKERS_LOAD_AUGMENTATION,
                                                max_queue_size=60)

        val_gen = ContinuousParallelBatchIterator(generator_val, ordered=False,
                                                batch_size=1,
                                                multiprocess=P.MULTIPROCESS_LOAD_AUGMENTATION,
                                                n_producers=P.N_WORKERS_LOAD_AUGMENTATION,
                                                max_queue_size=30)               

        train_gen.append(X_train)
        val_gen.append(X_val)

        logging.info("Starting training...")
        for epoch in range(P.N_EPOCHS):
            self.pre_epoch()

            if epoch in LR_SCHEDULE:
                logging.info("Setting learning rate to {}".format(LR_SCHEDULE[epoch]))
                self.l_r.set_value(LR_SCHEDULE[epoch])

            self.do_batches(self.train_fn, train_gen, self.train_metrics)
            self.do_batches(self.val_fn, val_gen, self.val_metrics)
            self.post_epoch()



if __name__ == "__main__":
    train_generator, validation_generator = patch_sampling.prepare_sampler()

    X_train = [P.BATCH_SIZE_TRAIN]*(P.EPOCH_SAMPLES_TRAIN//P.BATCH_SIZE_TRAIN)*P.N_EPOCHS
    X_val = [P.BATCH_SIZE_VALIDATION]*(P.EPOCH_SAMPLES_VALIDATION//P.BATCH_SIZE_VALIDATION)*P.N_EPOCHS

    trainer = VGGFCNetTrainer()
    trainer.train(train_generator, X_train, validation_generator, X_val)