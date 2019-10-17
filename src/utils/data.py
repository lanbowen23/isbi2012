import numpy as np
from skimage.filters import gaussian
from skimage.transform import swirl, resize
from skimage.util import random_noise, crop
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from random import randint
import logging
from src.utils.runtime import funcname


def random_transforms(items, nb_min=0, nb_max=5, rng=np.random, aug=False):
    first_transforms = [
        lambda x: x,
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.rot90(x, 1),
        lambda x: np.rot90(x, 2),
        lambda x: np.rot90(x, 3),
    ]
    elastic_transforms = [
        lambda x: x,
        lambda x: add_elastic_transform(x, 1024, 512*0.08),
    ]
    val = 10
    second_transforms = [
        lambda x: x,
        lambda x: x,
        lambda x: x,
        lambda x: add_elastic_transform(x, 10),
        lambda x: add_elastic_transform(x, 20),
        lambda x: add_elastic_transform(x, 30),
        lambda x: add_gaussian_noise(x, 0, val),
        lambda x: add_uniform_noise(x, -val, val),
        lambda x: change_brightness(x, val),
    ]

    n = rng.randint(nb_min, nb_max + 1)
    items_t = [item.copy() for item in items]

    for _ in range(n):
        idx1  = rng.randint(0, len(first_transforms))
        transform1 = first_transforms[idx1]
        items_t = [transform1(item) for item in items_t]
    
    if aug:
        idx2, idx3 = rng.randint(0, len(second_transforms)), rng.randint(0, len(elastic_transforms))
        transform2, transform3 = second_transforms[idx2], elastic_transforms[idx3]
        items_t = [transform3(item) for item in items_t]
        items_t[0] = transform2(items_t[0])
    
    return items_t


def add_elastic_transform(image, alpha, sigma=3, pad_size=0, seed=None):
    """
    Args:
        image : numpy array of image
        alpha : α is a scaling factor, shift distance
        sigma : σ is an elasticity coefficient, gausian kernel std
        random_state = random integer
    Return :
        image : elastically transformed numpy array of image
    """
    if seed is None:
        seed = randint(1, 100)
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState(seed)
        
    image = np.pad(image, pad_size, mode="symmetric")
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),  # [-1, 1] uniform distribution
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))  # simulate coordinates
    coordinates = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    new_image = map_coordinates(image, coordinates, order=1).reshape(shape)
    return cropping(new_image, 512, pad_size, pad_size)

def normalization1(image, mean, std):
    """ Normalization using mean and std
    Args :
        image : numpy array of image
        mean :
    Return :
        image : numpy array of image with values turned into standard scores
    """

    image = image / 255  # values will lie between 0 and 1.
    image = (image - mean) / std

    return image

def normalization2(image, max=1, min=0):
    """Normalization to range of [min, max]
    Args :
        image : numpy array of image
        mean :
    Return :
        image : numpy array of image with values turned into standard scores
    """
    image_new = (image-np.min(image))*(max-min)/(np.max(image)-np.min(image)) + min
    return image_new

def add_gaussian_noise(image, mean=0, std=1):
    """
    Args:
        image : numpy array of image
        mean : pixel mean of image
        standard deviation : pixel standard deviation of image
    Return :
        image : numpy array of image with gaussian noise added
    """
    gaus_noise = np.random.normal(mean, std, image.shape)
    image = image.astype("int16")
    noise_img = image + gaus_noise
    image = ceil_floor_image(image)
    return noise_img

def add_uniform_noise(image, low=-10, high=10):
    """
    Args:
        image : numpy array of image
        low : lower boundary of output interval
        high : upper boundary of output interval
    Return :
        image : numpy array of image with uniform noise added
    """
    uni_noise = np.random.uniform(low, high, image.shape)
    image = image.astype("int16")
    noise_img = image + uni_noise
    image = ceil_floor_image(image)
    return noise_img

def change_brightness(image, value):
    """
    Args:
        image : numpy array of image
        value : brightness
    Return :
        image : numpy array of image with brightness added
    """
    image = image.astype("int16")
    image = image + value
    image = ceil_floor_image(image)
    return image

def ceil_floor_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image

def cropping(image, crop_size, dim1, dim2):
    """crop the image and pad it to in_size
    Args :
        images : numpy array of images
        crop_size(int) : size of cropped image
        dim1(int) : vertical location of crop
        dim2(int) : horizontal location of crop
    Return :
        cropped_img: numpy array of cropped image
    """
    cropped_img = image[dim1:dim1+crop_size, dim2:dim2+crop_size]
    return cropped_img