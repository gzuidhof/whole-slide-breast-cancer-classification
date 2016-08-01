import numpy as np
import scipy.misc

input_folder = "../data/*"
files = glob.glob(input_folder)
files = filter(lambda x: 'mask' not in x, files)

images = []

for f in files:
    images.append(scipy.misc.imload(f))
    

images = np.array(images)
print np.mean(images






