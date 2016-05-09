import numpy as np

def float32(k):
    return np.cast['float32'](k)

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
        from http://goo.gl/DZNhk
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def normalize_image(namedNdArray):
    """ Normalization function to map RGB range to 0 - 1. """
    namedNdArray.data = (namedNdArray.data / 255.)
    return namedNdArray    
    
def random_flips(inputs):
    bs = inputs.shape[0]
    indices = np.random.choice(bs, bs / 2, replace=False)
         
    RandomOnes1 = np.random.choice([0, 1], bs, replace=True)*2 - 1     
    RandomOnes2 = np.random.choice([0, 1], bs, replace=True)*2 - 1    
    for indice in indices:
        inputs[indice,:,:,:] = inputs[indice,:,::RandomOnes1[indice],::RandomOnes2[indice]]
    return inputs