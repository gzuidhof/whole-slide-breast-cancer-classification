import numpy as np
import multiresolutionimageinterface as mir

class WSISampler(object):
    """
    Sampler object, uses multiresolutionimageinterface for sampling and a custom mask format.

    
    """

    def __init__(self, filename, mask, data_level, patch_size):
        """
            filename string:path to the WSI image (generally a .tif)
            mask wsi_mask.WSIMask: mask object loaded with the mask of this image
            data_level: integer data level to load patches from
            patch_size: tuple with size of patches to extract (can be non-square)
        """
        self.filename = filename
        self.mask = mask
        self.data_level = data_level
        self.mask = mask
        self.patch_size = patch_size
        self.dims = None
        self.labels = self.mask.labels

        assert data_level > -1
        assert len(patch_size) == 2
        
    def sample(self, return_position=False, label=None):
        x,y = self.mask.generate_position(label)
        
        r = mir.MultiResolutionImageReader()
        img = r.open(self.filename)

        if self.dims is None:
            self.dims = img.getLevelDimensions(self.data_level)

        image = img.getUCharPatch(x,y, self.patch_size[0], self.patch_size[1], self.data_level)
        img.close()
        image = image.transpose(2,0,1) #From 0,1,c to c,0,1

        if return_position:
            return image, (x,y)

        return image
        

    def sample_full(self):
        r = mir.MultiResolutionImageReader()
        img = r.open(self.filename)
        if self.dims is None:
            self.dims = img.getLevelDimensions(self.data_level)
        
        image = img.getUCharPatch(0,0, self.dims[0], self.dims[1], self.data_level)
        return image.transpose(2,0,1)  #From 0,1,c to c,0,1

        



    