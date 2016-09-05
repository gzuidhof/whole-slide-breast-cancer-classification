import numpy as np
import multiresolutionimageinterface as mir
from scipy import ndimage

class WSISampler(object):
    """
    Sampler object, uses multiresolutionimageinterface for sampling and a custom mask format.
    To create it you need a corresponding WSIMask object.
    """

    def __init__(self, filename, mask, data_level, patch_size, cache_size=1):
        """
            filename string:path to the WSI image (generally a .tif)
            mask wsi_mask.WSIMask: mask object loaded with the mask of this image
            data_level: integer data level to load patches from
            patch_size: tuple with size of patches to extract (can be non-square, although not tested)
            cache_size: integer amount of patches to extract per file open (larger generally means faster, but more memory usage)
        """
        self.filename = filename
        self.is_multires = '.tif' in self.filename
        self.mask = mask
        self.data_level = data_level
        self.mask = mask
        self.patch_size = patch_size
        self.dims = None
        self.labels = self.mask.labels

        self.cache = []
        self.cache_size = cache_size
        self.last_asked_label = None

        if data_level != mask.data_level: # This is not supported (it is certainly possible though).
            print "Non-matching data level for mask and sample ({})!".format(filename)

        assert data_level > -1
        assert len(patch_size) == 2
        assert cache_size > 0
        
    def sample(self, label=None):
        """
            Sample a patch (optionally with given label)
        """

        # Time to fill the cache?
        if len(self.cache) == 0 or self.last_asked_label != label:

            self.cache = []
            self.last_asked_label = label

            if self.is_multires:
                self.fill_cache_mir(label)
            else:
                self.fill_cache_non_mir(label)

        # Take from cached samples
        return self.cache.pop()
        
    def fill_cache_mir(self, label=None):
            r = mir.MultiResolutionImageReader()
            img = r.open(self.filename)

            # Fill the cache
            for i in range(self.cache_size):

                x,y = self.mask.generate_position(label)
                
                #if self.dims is None: #Lazily load level dimensions
                #    self.dims = img.getLevelDimensions(self.data_level)

                image = img.getUCharPatch(x,y, self.patch_size[0], self.patch_size[1], self.data_level)
                image = image.transpose(2,0,1) #From 0,1,c to c,0,1
                
                self.cache.append(image)
            
            img.close()

    def fill_cache_non_mir(self, label=None):
        img = ndimage.imread(self.filename)

        for i in range(self.cache_size):

                x,y = self.mask.generate_position(label)

                image = img[x:x+self.patch_size[0], y:y+self.patch_size[1], :] 
                image = image.transpose(2,0,1) #From 0,1,c to c,0,1
                
                self.cache.append(image)
        
        del img

    def sample_full(self):
        """
            Load the full WSI image into a numpy array (c01 channel order, ready for convnets)
        """

        if self.is_multires:
            r = mir.MultiResolutionImageReader()
            img = r.open(self.filename)
            if self.dims is None:
                self.dims = img.getLevelDimensions(self.data_level)
            
            image = img.getUCharPatch(0,0, self.dims[0], self.dims[1], self.data_level)
        else: 
            image = ndimage.imread(self.filename)
        return image.transpose(2,0,1)  #From 0,1,c to c,0,1

        



    