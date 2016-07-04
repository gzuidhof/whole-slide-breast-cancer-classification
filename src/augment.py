from __future__ import division
from params import params as P
import numpy as np


try:
    import cv2
    CV2_AVAILABLE=True
    print "OpenCV 2 available, using that for augmentation"
    from scipy.ndimage.interpolation import rotate, shift, zoom, affine_transform
    from skimage.transform import warp, AffineTransform
except:
    from scipy.ndimage.interpolation import rotate, shift, zoom, affine_transform
    from skimage.transform import warp, AffineTransform
    CV2_AVAILABLE=False
    print "OpenCV 2 NOT AVAILABLE, using skimage/scipy.ndimage instead"

def augment(images):
    pixels = images[0].shape[1]
    center = pixels/2.-0.5

    random_flip_x = P.AUGMENTATION_PARAMS['flip'] and np.random.randint(2) == 1
    random_flip_y = P.AUGMENTATION_PARAMS['flip'] and np.random.randint(2) == 1

    # Translation shift
    shift_x = np.random.uniform(*P.AUGMENTATION_PARAMS['translation_range'])
    shift_y = np.random.uniform(*P.AUGMENTATION_PARAMS['translation_range'])
    rotation_degrees = int(np.random.uniform(*P.AUGMENTATION_PARAMS['rotation_range']))
    zoom_factor = np.random.uniform(*P.AUGMENTATION_PARAMS['zoom_range'])

    if CV2_AVAILABLE:
        M = cv2.getRotationMatrix2D((center, center), rotation_degrees, zoom_factor)
        M[0, 2] += shift_x
        M[1, 2] += shift_y

    for i in range(len(images)):
        image = images[i]

        if CV2_AVAILABLE:
            image = cv2.warpAffine(image, M, (pixels, pixels))
            if random_flip_x:
                image = cv2.flip(image, 0)
            if random_flip_y:
                image = cv2.flip(image, 1)
        else:

            if rotation_degrees > 0:
                rotate(image, rotation_degrees, reshape=False, output=image)
            #image = zoom(image, [1,zoom_factor,zoom_factor])
            if shift_x > 0 or shift_y > 0:
                shift(image, [0,shift_x,shift_y], output=image)
            
            if random_flip_x:
                image = flip_axis(image, 1)
            if random_flip_y:
                image = flip_axis(image, 2)

        images[i] = image

    images = crop(images, P.INPUT_SIZE)
    return images

def crop(images, desired_size):
    offset = (images.shape[2] - desired_size) // 2
    
    if offset > 0:
        return images[:,:,offset:-offset,offset:-offset]
    return images


def crop_or_pad(image, desired_size, pad_value):
    if image.shape[1] < desired_size:
        offset = (desired_size-image.shape[1])//2
        image = np.pad(image, ((0,0),(offset,offset),(offset,offset)), 'constant', constant_values=pad_value)
        if image.shape[1] != desired_size:
            new_image = np.full((image.shape[0],image.shape[1]+1,image.shape[1]+1),fill_value=pad_value)
            new_image[:,:image.shape[1],:image.shape[1]]=image
            image = new_image
    if image.shape[1] > desired_size:
        offset = (image.shape[1]-desired_size)//2
        image = image[:,offset:offset+desired_size,offset:offset+desired_size]

    return image


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x
