import numpy as np
import multiresolutionimageinterface as mir

class WSIMask(object):

    def __init__(self, filename, labels=[], mask_image=None, border_distance=0, data_level=0):
        self.filename = filename
        self.image = mask_image

        if self.border_distance 
        self.border_distance = border_distance
        self.data_level = data_level
        self.image = mask_image

        # Label that is to be sampled from this image
        self.labels = labels
        if self.image is None:
            self.load()

    def load(self):

        r = mir.MultiResolutionImageReader()
        img = r.open(self.filename)
        dims = img.getLevelDimensions(self.data_level)
        bd = self.border_distance
        image = img.getUCharPatch(bd, bd, dims[0]-bd, dims[1]-bd, self.data_level)

        image = image.transpose(2,0,1) #From 0,1,c to c,0,1
        image = image[0] #Take the first channel, the only channel
        img.close()

        # Non-black pixel indices
        coords = np.argwhere(image)
        
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1

        # Crop image
        self.image = image[x0:x1, y0:y1]
        self.offset = (bd+x0, bd+y0)

        self.image = self.image.astype(np.int8)

        # There is at least one pixel we can sample from
        assert np.sum(image) > 0

        if len(self.labels) > 0:
            # Label is present in image
            assert self.labels in np.unique(image).astype(np.int8)
        else:
            self.labels = np.unique(image)[1:]

    def generate_position(self, label=None):
        tries = 0

        while tries < 3000:
        
            x = np.random.randint(self.image.shape[0])
            y = np.random.randint(self.image.shape[1])

            if (label is None and self.image[x,y] in self.labels) or self.image[x,y] in self.labels:
                # NOTE THAT X AND Y ARE REVERSED, sampling with MIR has different ordering of images!!!
                return y + self.offset[1], x + self.offset[0]
            
            tries += 1

        raise Exception("Failed to extract patch after 3000 tries")










    