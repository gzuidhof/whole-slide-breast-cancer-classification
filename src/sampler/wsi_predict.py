from __future__ import division
import os
import numpy as np
import theano
import theano.tensor as T
import lasagne
import sys
sys.path.append("../")
import params

MODEL_PATH = '../../models/1472001110_stacked/1472001110_stacked_epoch224.npz'
params.params = params.Params(['../../config/default.ini'] + 
                              ['../../config/stacked.ini', '../../config/titania_stacked.ini'])

BACKGROUND_DATA_LEVEL = 3

SUPER_PATCH_SIZE=2048
PATCH_SIZE = 768
STRIDE = 128
DATA_LEVEL = 1
BATCH_SIZE = 48

WSI_PATH = '/mnt/rdstorage1/Userdata/Guido/slides/T13-37-I9-1.mrxs'#/mnt/rdstorage1/Userdata/Guido/slides/T10-11269-I-7-1.mrxs'

from params import params as P
import resnet
from wsi_parallel_sampler import WSIParallelSampler
import itertools
import util
from tqdm import tqdm
import pickle


def load_model(model_path):
    input_var = T.tensor4('inputs')

    net = resnet.ResNet_FullPre_Wide(input_var, 4, 2)
    all_layers = lasagne.layers.get_all_layers(net)
    net = all_layers[-3]
    net = resnet.ResNet_Stacked_Old(net)

    with np.load(model_path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    lasagne.layers.set_all_param_values(net, param_values)
    del param_values

    print "Compiling predict function"
    predict_fn = resnet.define_predict(net, input_var)

    return predict_fn


def generate_patch_positions(xlim, ylim, patch_size, stride, data_level):
    """
        Generates X and Y positions to sample at given a stride and data level
        They are ordered by Y first for better cache locality (=Fortran ordering)
    """
    positions = []
    
    scale_factor = 2**data_level
    
    x_positions = np.arange(0, xlim, stride*scale_factor)
    y_positions = np.arange(0, ylim, stride*scale_factor)

    # First Y, then X. Faster due to cache locality.
    for y in y_positions:
        for x in x_positions:
            positions.append( (x,y, patch_size, patch_size) )
            
            
    prediction_shape = np.array((len(x_positions), len(y_positions)))
    return positions, prediction_shape


def determine_background(image_dims, data_level):
    """
        Determine which pixels are fully background at a certain data level.
    """
    sampler = WSIParallelSampler(WSI_PATH, data_level=data_level, multiprocess=False, n_producers=1)
    positions = [(0, 0, image_dims[0]//2**data_level, image_dims[1]//2**data_level)]
    sampler.set_todo_positions(positions)

    imgs = list(sampler)
    mask = np.array(np.sum(np.array(imgs),axis=(0,1))>0,dtype=np.uint8)

    return mask


def indices_full_black(positions, mask, image_dims):
    """
        Determines which patches are completely background (which is likely in a lower resolution)
    """
    indices = []
    mask = mask.transpose(1,0)
    scale_factor = int(np.round(image_dims[0]/mask.shape[0]))
    
    for i, pos in enumerate(positions):
        pos = (np.array(pos)/scale_factor).astype(np.int32)
        if np.sum(mask[pos[0]:pos[0]+pos[2], pos[1]:pos[1]+pos[2]]) == 0:
            indices.append(i)

    bg_mask = np.zeros((len(positions,)))
    bg_mask[indices] = 1
    #Flattened mask
    return indices, bg_mask

def non_background_patch_generator(n, source_generator, bg_mask):
    """
        Only returns patches that are non-background. Skips others.
    """
    it = iter(source_generator)
    
    for i in xrange(n):
        if bg_mask[i]==1:
            continue
        else:
            yield (i, next(it))

def overlapping_patch_generator(sampler, positions, patch_size_super, data_level):
    new_positions = []
    
    scale_factor = 2**data_level
    
    # Determine the original locations in the WSI (account for data level)
    for pos in positions:
        new_positions.append( (pos[0]//scale_factor, pos[1]//scale_factor, pos[2], pos[3]) )
    
    positions = new_positions
    
    patch_size = positions[0][2]
    
    positions_to_sample = []
    positions_in_super = []

    offsets = [None]*len(positions)
        
    # Indices we still have to account for in super patches
    positions_todo = range(len(positions))
    
    
    print "Generating super patches (for positions N", len(positions), ")"
    
    while len(positions_todo) > 0:
        
        start_pos_index = positions_todo.pop(0)
        
        #Positions captured in super patch
        positions_in_current = []
        positions_in_current.append(start_pos_index)
        offsets[start_pos_index] = (0,0)
        
        
        start_pos = positions[start_pos_index]
        x = start_pos[0] #X of super patch
        y = start_pos[1]
        
        x_max = x + patch_size_super
        y_max = y + patch_size_super
        
        x_patch_size = patch_size
        y_patch_size = patch_size
            
        for p_index in positions_todo:
            p = positions[p_index]
            
            # Patch fits in max size of larger patch
            if ((p[0] >= x and p[0]+patch_size <= x_max) and
                (p[1] >= y and p[1]+patch_size <= y_max)):
            
                # Update to larger patch size if necessary
                x_patch_size = max(x_patch_size, p[0]+patch_size-x)
                y_patch_size = max(y_patch_size, p[1]+patch_size-y)  
                

                offsets[p_index] = (p[0]-x, p[1]-y)
                positions_in_current.append(p_index)
                
            # Because the positions are in order we can stop looking here.
            if (p[0]+patch_size > x_max and
                p[1]+patch_size > y_max):
                break
                
                
                
        positions_to_sample.append( (x*scale_factor, y*scale_factor, x_patch_size, y_patch_size) )
        positions_in_super.append(positions_in_current)
        positions_todo = [index for index in positions_todo if index not in positions_in_current]
        
    
    print "N positions to sample", len(positions_to_sample)

    
    # Make some assertions to make sure all positions are valid
    for p_indices, sample_pos in zip(positions_in_super, positions_to_sample)[5:6]:
        for p in p_indices:
            try:
                assert sample_pos[2] - offsets[p][0] >= patch_size
                assert sample_pos[3] - offsets[p][1] >= patch_size
            except:
                print "Assertion failed for position index ", p
            
    
    
    sampler.set_todo_positions(positions_to_sample)
    
    it = iter(sampler)
    
    ready_patches = [None]*len(positions)

    def gen():
        for p_index in xrange(len(positions)):


            if ready_patches[p_index] is None:

                super_patch = next(it)
                indices = positions_in_super.pop(0)

                for i in indices:
                    offset = offsets[i]
                    
                    # X and Y are reversed here!!! Strange ordering in MIR
                    patch = super_patch[:, offset[1]:offset[1]+patch_size, offset[0]:offset[0]+patch_size]
                        
                    ready_patches[i] = patch
                    
                    
            yield ready_patches[p_index]

            # Unload from memory
            ready_patches[p_index] = None
            
    return gen()



if __name__ == "__main__":

    sampler = WSIParallelSampler(WSI_PATH, data_level=0, multiprocess=False, n_producers=1)
    image_dims = sampler.get_image_dimensions()
    del sampler
    
    print "Image dimensions:", image_dims
    print "Determining lowres background mask"
    lowres_bg_mask = determine_background(image_dims, BACKGROUND_DATA_LEVEL)


    print "Loading model"
    predict_fn = load_model(MODEL_PATH)
    sampler = WSIParallelSampler(WSI_PATH, data_level=DATA_LEVEL, multiprocess=True, n_producers=8, max_queue_size=16)

    print "Generating positions to predict."
    positions, prediction_shape = generate_patch_positions(image_dims[0], image_dims[1], PATCH_SIZE, STRIDE, DATA_LEVEL)

    print "Determining which positions are fully background."
    background_indices, bg_mask = indices_full_black(positions, lowres_bg_mask, image_dims)
    non_background_positions = [positions[p] for p in range(len(positions)) if bg_mask[p] == 0]
    n_patches = len(positions)
    

    print "Image size", image_dims
    print "Amount of patches to predict: ", n_patches
    print "Amount of black patches:", len(background_indices), "N remaining", len(non_background_positions)
    print "Prediction shape", prediction_shape

    assert n_patches == np.product(prediction_shape)
    assert len(background_indices)+len(non_background_positions) == n_patches

    # Generator that works by sampling larger patches, to then extract the original patch size out of it.
    # Especially with a small stride this makes things much more efficient as there is a huge overlap otherwise.
    super_generator = overlapping_patch_generator(sampler, non_background_positions, patch_size_super=SUPER_PATCH_SIZE, data_level=DATA_LEVEL)
    non_bg_super_generator = non_background_patch_generator(len(positions), super_generator, bg_mask)

    
    n_batches = int(np.ceil(n_patches/BATCH_SIZE))
    n_non_background = n_patches - len(background_indices)

    prediction_flat = np.zeros((np.product(prediction_shape),)+(3,))

    images_in_batch = []
    indices_in_batch = []

    print "Starting prediction"

    for n, im in tqdm(non_bg_super_generator, total=n_non_background, smoothing=0.1):   
    
        images_in_batch.append(im)
        indices_in_batch.append(n)
            
        if len(indices_in_batch) == BATCH_SIZE or n == n_non_background-1:
            image_batch = np.array(images_in_batch, dtype=np.float32)
        else:
            continue

        image_batch = util.normalize_image(image_batch)
        image_batch = util.zero_center(image_batch, P.MEAN_PIXEL).astype(np.float32)
        pred_binary, pred = predict_fn(image_batch)
        
        prediction_flat[indices_in_batch] = pred
        
        images_in_batch = []
        indices_in_batch = []

    slide_filename = WSI_PATH.split('/')[-1].split('.')[0] + "_" + str(STRIDE)
    save_filename = '../../wsi_predictions/' + slide_filename + '.npy'
    plot_filename = '../../wsi_predictions/' + slide_filename + '.png'

    prediction = prediction_flat.reshape(tuple(prediction_shape)+(3,), order='F')
    np.save(save_filename, prediction)

    import matplotlib.pyplot as plt
    plt.imsave(plot_filename, prediction.transpose(1,0,2))
        