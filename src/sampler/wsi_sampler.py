import numpy as np
import multiresolutionimageinterface as mir

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
        """
        self.filename = filename
        self.mask = mask
        self.data_level = data_level
        self.mask = mask
        self.patch_size = patch_size
        self.dims = None
        self.labels = self.mask.labels

        self.cache = []
        self.cache_size = cache_size


        if data_level != mask.data_level:
            print "Non-matching data level for mask and sample ({})!".format(filename)

        assert data_level > -1
        assert len(patch_size) == 2
        assert cache_size > 0
        
    def sample(self, label=None):

        # Time to fill the cache
        if len(self.cache) == 0:
            r = mir.MultiResolutionImageReader()
            img = r.open(self.filename)

            # Fill the cache
            for i in range(self.cache_size):

                x,y = self.mask.generate_position(label)
                
                if self.dims is None:
                    self.dims = img.getLevelDimensions(self.data_level)

                image = img.getUCharPatch(x,y, self.patch_size[0], self.patch_size[1], self.data_level)
                image = image.transpose(2,0,1) #From 0,1,c to c,0,1
                
                self.cache.append(image)
            
            img.close()


        # Take from cached samples
        return self.cache.pop()
        
    def sample_full(self):
        """
            Load the full WSI image into a numpy array (c01 channel order, ready for convnets)
        """
        r = mir.MultiResolutionImageReader()
        img = r.open(self.filename)
        if self.dims is None:
            self.dims = img.getLevelDimensions(self.data_level)
        
        image = img.getUCharPatch(0,0, self.dims[0], self.dims[1], self.data_level)
        return image.transpose(2,0,1)  #From 0,1,c to c,0,1

        



    