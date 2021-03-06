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
from glob import glob

if __name__ == "__main__":
    import theano
    import theano.tensor as T
    import lasagne
    import resnet
    import patch_sampling
    from dataset import label_name


models_folder = '../models/'
model_instances = sorted(glob(models_folder+'*'), reverse=True)

def get_model_to_stack_on():

    model_name = 'resnet_2class' if P.STACK_ON_N_CLASSES == 2 else 'resnet_3class'

    for m in model_instances:
        if model_name in m:
            print "Model ", model_name, "using folder", m
            m_name = m.split('/')[-1]

            epoch_120_path = m+'/'+m_name+"_epoch120.npz"
            print epoch_120_path

            if len (glob(epoch_120_path)) > 0:
                print "Found model", epoch_120_path
            else:
                continue

            best_epoch_saves = glob(m+'/'+m_name+"_best_epoch*.npz")
            epoch_save_numbers = map(lambda x: int(x.split('epoch')[-1].split('.npz')[0]), best_epoch_saves)
            weight_save_file = best_epoch_saves[ np.argmax(epoch_save_numbers) ]

            logging.info("Stacking on top of "+ weight_save_file)
            return weight_save_file

            #return pd.DataFrame.from_csv(os.path.join(models_folder, m, 'metrics.csv'))




class ResNetTrainer(trainer.Trainer):
    def __init__(self):
        metric_names = ['Loss','L2','Accuracy', 'BinaryAccuracy']
        super(ResNetTrainer, self).__init__(metric_names)

        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

        logging.info("Defining network")
        net = resnet.ResNet_FullPre_Wide(input_var, P.DEPTH, P.BRANCHING_FACTOR)
        #net=resnet.ResNet_FullPreActivation(input_var, P.DEPTH)
        all_layers = lasagne.layers.get_all_layers(net)

        print "Loading model"
        #model_save_file = '../models/wide_resnet_babak.npz'#'../models/1470945732_resnet/1470945732_resnet_epoch478.npz'
        #model_save_file = '../models/1473540491_resnet/1473540491_resnet_epoch62.npz'

        model_save_file = get_model_to_stack_on()


        logging.info("Model file "+ model_save_file)

        with np.load(model_save_file) as f:
        	param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        
        # Make up some weights
        if len(param_values[-1]) == 2 and P.N_CLASSES == 3:

            cl3_weight = np.zeros((1,),dtype=np.float32)
            param_values[-1] = np.hstack((param_values[-1], cl3_weight))

            cl3_weights = np.zeros((len(param_values[-2]), 1), dtype=np.float32)
            param_values[-2] = np.hstack((param_values[-2], cl3_weights))
        
        #print len(param_values)

        lasagne.layers.set_all_param_values(net, param_values)
        del param_values



        # New output layer
        net = all_layers[-3] #lasagne.layers.get_output_shape(all_layers[-3])


        #for x in all_layers[:-3]:
        #    print x.output_shape

        # Freeze the layers
        for layer in lasagne.layers.get_all_layers(net):
            for param in layer.params:
                layer.params[param].discard('trainable')

        net = resnet.ResNet_Stacked(net)

        print "Compiling network"
        self.network = net
        train_fn, val_fn, l_r = resnet.define_updates(self.network, input_var, target_var)

        self.train_fn = train_fn
        self.val_fn = val_fn
        self.l_r = l_r

    def save_debug_images(self, n, images, labels, metrics):
        im = images.transpose(0,2,3,1)

        im[:,:,:,0] += P.MEAN_PIXEL[0]
        im[:,:,:,1] += P.MEAN_PIXEL[1]
        im[:,:,:,2] += P.MEAN_PIXEL[2]

        f, axarr = plt.subplots(4,4,figsize=(12,12))
        for i in range(min(16, len(images))):
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
        
        #batch_size = P.EPOCH_SAMPLES_TRAIN//P.BATCH_SIZE_TRAIN if metrics.name=='train' else P.EPOCH_SAMPLES_VALIDATION//P.BATCH_SIZE_VALIDATION
        batch_size = P.EPOCH_SAMPLES_TRAIN if metrics.name=='train' else P.EPOCH_SAMPLES_VALIDATION

        for i, batch in enumerate(tqdm(batch_generator(batch_size))):
            inputs, targets, filenames = batch

            err, l2_loss, acc, prediction, _ = fn(inputs, targets)

            predictions_binary_problem = np.where(prediction>0,1,0)
            targets_binary_problem = np.where(targets>0,1,0)
            
            binary_accuracy = np.sum(predictions_binary_problem==targets_binary_problem)/np.product(prediction.shape)


            metrics.append([err, l2_loss, acc, binary_accuracy])
            metrics.append_prediction(targets, prediction)

            if i == 0:
                self.save_debug_images(i, inputs, targets, metrics)

    def train(self, generator_train, X_train, generator_val, X_val):

        train_gen = ContinuousParallelBatchIterator(generator_train, ordered=False,
                                                batch_size=1,
                                                multiprocess=P.MULTIPROCESS_LOAD_AUGMENTATION,
                                                n_producers=P.N_WORKERS_LOAD_AUGMENTATION,
                                                max_queue_size=min([8, P.EPOCH_SAMPLES_TRAIN*2]))

        val_gen = ContinuousParallelBatchIterator(generator_val, ordered=False,
                                                batch_size=1,
                                                multiprocess=P.MULTIPROCESS_LOAD_AUGMENTATION,
                                                n_producers=P.N_WORKERS_LOAD_AUGMENTATION,
                                                max_queue_size=min([6, P.EPOCH_SAMPLES_VALIDATION*2]))       

        train_gen.append(X_train)
        val_gen.append(X_val)

        logging.info("Learning rate set to {}".format(P.LEARNING_RATE))
        self.l_r.set_value(P.LEARNING_RATE)

        logging.info("Starting training...")
        for epoch in range(P.N_EPOCHS):
            self.pre_epoch()

            self.do_batches(self.train_fn, train_gen, self.train_metrics)
            self.do_batches(self.val_fn, val_gen, self.val_metrics)
            self.post_epoch()

if __name__ == "__main__":
    # Load the samplers (which loads the masks)
    
    X_train = [P.BATCH_SIZE_TRAIN]*(P.EPOCH_SAMPLES_TRAIN)*P.N_EPOCHS
    X_val = [P.BATCH_SIZE_VALIDATION]*(P.EPOCH_SAMPLES_VALIDATION)*P.N_EPOCHS
    
    trainer = ResNetTrainer()
    train_generator, validation_generator = patch_sampling.prepare_custom_sampler(mini_subset=False, override_cache_size=1)
    trainer.train(train_generator, X_train, validation_generator, X_val)