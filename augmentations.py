from scipy.ndimage import gaussian_filter
from skimage import transform as sk_tf
from collections import namedtuple
import numpy as np
import numbers


def interval(obj, lower=None):
    """ Listify an object.

    Parameters
    ----------
    obj: 2-uplet or number
        the object used to build the interval.
    lower: number, default None
        the lower bound of the interval. If not specified, a symetric
        interval is generated.

    Returns
    -------
    interval: 2-uplet
        an interval.
    """
    if isinstance(obj, numbers.Number):
        if obj < 0:
            raise ValueError("Specified interval value must be positive.")
        if lower is None:
            lower = -obj
        return (lower, obj)
    if len(obj) != 2:
        raise ValueError("Interval must be specified with 2 values.")
    min_val, max_val = obj
    if min_val > max_val:
        raise ValueError("Wrong interval boudaries.")
    return tuple(obj)


class Transformer(object):
    """ Class that can be used to register a sequence of transformations.
    """
    Transform = namedtuple("Transform", ["transform", "probability"])

    def __init__(self):
        """ Initialize the class.
        """
        self.transforms = []

    def register(self, transform, probability=1):
        """ Register a new transformation.
        Parameters
        ----------
        transform: callable
            the transformation object.
        probability: float, default 1
            the transform is applied with the specified probability.
        """
        trf = self.Transform(transform=transform, probability=probability, )
        self.transforms.append(trf)

    def __call__(self, arr):
        """ Apply the registered transformations.
        """
        transformed = arr.copy()
        for trf in self.transforms:
            if np.random.rand() < trf.probability:
                transformed = trf.transform(transformed)
        return transformed

    def __str__(self):
        if len(self.transforms) == 0:
            return '(Empty Transformer)'
        s = 'Composition of:'
        for trf in self.transforms:
            s += '\n\t- '+trf.__str__()
        return s


class Normalize(object):
    def __init__(self, mean=0.0, std=1.0, eps=1e-8):
        self.mean=mean
        self.std=std
        self.eps=eps

    def __call__(self, arr):
        return self.std * (arr - np.mean(arr))/(np.std(arr) + self.eps) + self.mean

class Crop(object):
    """Crop the given n-dimensional array either at a random location or centered"""
    def __init__(self, shape, type="center", resize=False, keep_dim=False):
        """:param
        shape: tuple or list of int
            The shape of the patch to crop
        type: 'center' or 'random'
            Whether the crop will be centered or at a random location
        resize: bool, default False
            If True, resize the cropped patch to the inital dim. If False, depends on keep_dim
        keep_dim: bool, default False
            if True and resize==False, put a constant value around the patch cropped. If resize==True, does nothing
        """
        assert type in ["center", "random"]
        self.shape = shape
        self.copping_type = type
        self.resize=resize
        self.keep_dim=keep_dim

    def __call__(self, arr):
        assert isinstance(arr, np.ndarray)
        assert type(self.shape) == int or len(self.shape) == len(arr.shape), "Shape of array {} does not match {}".\
            format(arr.shape, self.shape)

        img_shape = np.array(arr.shape)
        if type(self.shape) == int:
            size = [self.shape for _ in range(len(self.shape))]
        else:
            size = np.copy(self.shape)
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.copping_type == "center":
                delta_before = (img_shape[ndim] - size[ndim]) / 2.0
            elif self.copping_type == "random":
                delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(int(delta_before), int(delta_before + size[ndim])))
        if self.resize:
            # resize the image to the input shape
            return sk_tf.resize(arr[tuple(indexes)], img_shape, preserve_range=True)

        if self.keep_dim:
            mask = np.zeros(img_shape, dtype=np.bool)
            mask[tuple(indexes)] = True
            arr_copy = arr.copy()
            arr_copy[~mask] = 0
            return arr_copy

        return arr[tuple(indexes)]


class Cutout(object):
    """Apply a cutout on the images
    cf. Improved Regularization of Convolutional Neural Networks with Cutout, arXiv, 2017
    We assume that the square to be cut is inside the image.
    """
    def __init__(self, patch_size=None, value=0, random_size=False, inplace=False, localization=None):
        self.patch_size = patch_size
        self.value = value
        self.random_size = random_size
        self.inplace = inplace
        self.localization = localization

    def __call__(self, arr):

        img_shape = np.array(arr.shape)
        if type(self.patch_size) == int:
            size = [self.patch_size for _ in range(len(img_shape))]
        else:
            size = np.copy(self.patch_size)
        assert len(size) == len(img_shape), "Incorrect patch dimension."
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.random_size:
                size[ndim] = np.random.randint(0, size[ndim])
            if self.localization is not None:
                delta_before = max(self.localization[ndim] - size[ndim]//2, 0)
            else:
                delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(int(delta_before), int(delta_before + size[ndim])))
        if self.inplace:
            arr[tuple(indexes)] = self.value
            return arr
        else:
            arr_cut = np.copy(arr)
            arr_cut[tuple(indexes)] = self.value
            return arr_cut

class Flip(object):
    """ Apply a random mirror flip."""
    def __init__(self, axis=None):
        '''
        :param axis: int, default None
            apply flip on the specified axis. If not specified, randomize the
            flip axis.
        '''
        self.axis = axis

    def __call__(self, arr):
        if self.axis is None:
            axis = np.random.randint(low=0, high=arr.ndim, size=1)[0]
        return np.flip(arr, axis=(self.axis or axis))


class Blur(object):
    def __init__(self, snr=None, sigma=None):
        """ Add random blur using a Gaussian filter.
            Parameters
            ----------
            snr: float, default None
                the desired signal-to noise ratio used to infer the standard deviation
                for the noise distribution.
            sigma: float or 2-uplet
                the standard deviation for Gaussian kernel.
        """
        if snr is None and sigma is None:
            raise ValueError("You must define either the desired signal-to noise "
                             "ratio or the standard deviation for the noise "
                             "distribution.")
        self.snr = snr
        self.sigma = sigma

    def __call__(self, arr):
        sigma = self.sigma
        if self.snr is not None:
            s0 = np.std(arr)
            sigma = s0 / self.snr
        sigma = interval(sigma, lower=0)
        sigma_random = np.random.uniform(low=sigma[0], high=sigma[1], size=1)[0]
        return gaussian_filter(arr, sigma_random)


class Noise(object):
    def __init__(self, snr=None, sigma=None, noise_type="gaussian"):
        """ Add random Gaussian or Rician noise.

           The noise level can be specified directly by setting the standard
           deviation or the desired signal-to-noise ratio for the Gaussian
           distribution. In the case of Rician noise sigma is the standard deviation
           of the two Gaussian distributions forming the real and imaginary
           components of the Rician noise distribution.

           In anatomical scans, CNR values for GW/WM ranged from 5 to 20 (1.5T and
           3T) for SNR around 40-100 (http://www.pallier.org/pdfs/snr-in-mri.pdf).

           Parameters
           ----------
           snr: float, default None
               the desired signal-to noise ratio used to infer the standard deviation
               for the noise distribution.
           sigma: float or 2-uplet, default None
               the standard deviation for the noise distribution.
           noise_type: str, default 'gaussian'
               the distribution of added noise - can be either 'gaussian' for
               Gaussian distributed noise, or 'rician' for Rice-distributed noise.
        """

        if snr is None and sigma is None:
            raise ValueError("You must define either the desired signal-to noise "
                             "ratio or the standard deviation for the noise "
                             "distribution.")
        assert noise_type in {"gaussian", "rician"}, "Noise muse be either Rician or Gaussian"
        self.snr = snr
        self.sigma = sigma
        self.noise_type = noise_type


    def __call__(self, arr):
        sigma = self.sigma
        if self.snr is not None:
            s0 = np.std(arr)
            sigma = s0 / self.snr
        sigma = interval(sigma, lower=0)
        sigma_random = np.random.uniform(low=sigma[0], high=sigma[1], size=1)[0]
        noise = np.random.normal(0, sigma_random, [2] + list(arr.shape))
        if self.noise_type == "gaussian":
            transformed = arr + noise[0]
        elif self.noise_type == "rician":
            transformed = np.square(arr + noise[0])
            transformed += np.square(noise[1])
            transformed = np.sqrt(transformed)
        return transformed


