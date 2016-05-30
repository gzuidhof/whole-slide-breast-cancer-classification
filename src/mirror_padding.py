from __future__ import division

import theano
import theano.tensor as T
import numpy as np

import scipy.misc
import matplotlib.pyplot as plt
import lasagne

class MirrorPadLayer(lasagne.layers.Layer):

    def __init__(self, incoming, width, batch_ndim=2, **kwargs):
        super(MirrorPadLayer, self).__init__(incoming, **kwargs)
        self.width = width
        self.batch_ndim = batch_ndim

    def get_output_for(self, input, **kwargs):
        return mirror_pad(input, self.width, self.batch_ndim)

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
    
        for dim in range(self.batch_ndim,len(input_shape)):
            output_shape[dim] = self.width[dim-self.batch_ndim]+input_shape[dim]
        
        return output_shape
    
def mirror_pad(input, width, batch_ndim=2):
    pad_amount = width
    input_ndim = input.ndim
    all_dims = [slice(None)]*input_ndim 
    
    new_shape = list(input.shape)
    
    center = list(all_dims)
    offset = [0]*input_ndim
    
    for dim in range(batch_ndim,input_ndim):
        padding = pad_amount[dim-batch_ndim]
        offset[dim] = padding//2
        new_shape[dim] = input.shape[dim]+offset[dim]*2
        
        # Location to place original image
        center[dim] = slice(offset[dim],offset[dim]+input.shape[dim])
            
    new_tensor = T.zeros(new_shape)
    # Place original tensor in the center
    new_tensor = T.inc_subtensor(new_tensor[center], input)
    
            
    for dim in range(batch_ndim,input_ndim):
        
        off = offset[dim]
    
        #Padding at the start of a dimension
        start_padding = list(all_dims)
        start_padding[dim] = slice(None,off,1)
        
        #Padding at the end of a dimension
        end_padding = list(all_dims)
        end_padding[dim] = slice(new_shape[dim]-off,None,1)
        
        #Sources of padding (reverse from the offset)
        start_source = list(all_dims) 
        start_source[dim] = slice(off*2-1,off-1,-1)
        
        end_source = list(all_dims)
        end_source[dim] = slice(new_shape[dim]-off-1,new_shape[dim]-off*2-1,-1)
        
        #Pad both ends of the dimensions 
        new_tensor = T.set_subtensor(new_tensor[start_padding], new_tensor[start_source])
        #continue
        #new_tensor = T.set_subtensor(new_tensor[end_padding], new_tensor[end_source])
    
    return new_tensor      
    
if __name__ == "__main__":
    
    test_image = np.array([[[[1,2,3],
              [4,5,6],
              [7,8,9]]]],dtype=np.float32)
              
    expected_image = np.array([[[[1, 1, 2, 3, 3],
                    [1, 1, 2, 3, 3],
                    [4, 4, 5, 6, 6],
                    [7, 7, 8, 9, 9],
                    [7, 7, 8, 9, 9]]]],dtype=np.float32)
    
    im = np.array(scipy.misc.imread('lena.jpg'),dtype=np.float32)/255
    im = im.transpose(2,0,1) #c, 0, 1
    im = np.array([im,im,im])
    #im = np.expand_dims(im.transpose(2,0,1),axis=0)
    #plt.imshow(np.squeeze(im).transpose(1,2,0))
    #plt.show()

    #z = T.
    input_var= T.ftensor4('inputs')

    #fn2 = mirror_pad(input_var, (2,2))

    #mirroring = mirror_pad(input_var, (256,256))

    #fn256 = theano.function([input_var], [mirroring])


    #out, = fn2(test_image)
    #assert(np.all(out==expected_image))

    x = lasagne.layers.InputLayer((3,3,256,256), input_var=input_var)
    
    lay = MirrorPadLayer(x,(2,4))
    out = lasagne.layers.get_output(lay)
    
    tr = theano.function([input_var],[out])

    lena_mirror = np.squeeze(tr(im)[0][1]).transpose(1,2,0)
    plt.imshow(lena_mirror)
    plt.show()
    
    
