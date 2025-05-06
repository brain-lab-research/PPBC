import random
import numpy as np
import cv2
import albumentations as A
from torchvision import transforms


def get_color_distortion(s=1.0):
    """It's taken from sicmlr"""
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class SobelFilter(A.ImageOnlyTransform):
    """Sobel filter"""

    def __init__(self, always_apply=False, p=0.2, ksize=31):
        super(SobelFilter, self).__init__(p, always_apply)
        self.ksize = ksize

    def apply(self, img, **params):
        # Convert image to grayscale
        ksize = random.choice([i for i in range(1, self.ksize) if i % 2 != 0])

        # Apply Sobel filter
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

        # Compute the magnitude of the gradient
        sobel_bgr = np.sqrt(sobel_x**2 + sobel_y**2)
        return sobel_bgr

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        # Example of how you can manipulate targets if needed
        # Here, we simply return an empty dict as Sobel filter does not depend on targets
        return {}


class ColorJitter(A.ColorJitter):
    """s=1"""

    def __init__(self, s=1.0, p=0.8, **kwargs):
        super().__init__(
            brightness=0.8 * s,
            contrast=0.8 * s,
            saturation=0.8 * s,
            hue=0.2 * s,
            p=p,
            **kwargs,
        )


class RandomGrayscale(A.ImageOnlyTransform):
    """
    Randomly convert an image to grayscale with a given probability.

    Args:
        p (float): Probability of applying the grayscale transformation.

    Targets:
        image
    """

    def __init__(self, p=0.2, always_apply=False):
        super(RandomGrayscale, self).__init__(always_apply)
        self.random_grayscale = transforms.RandomGrayscale(p=p)
        self.p = p

    def apply(self, image, **params):
        return self.random_grayscale(image)

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint


def color_distortion(s):
    """"""
    color_jitter = A.ColorJitter(
        brightness=0.8 * s,
        contrast=0.8 * s,
        saturation=0.8 * s,
        hue=0.2 * s,
        p=0.8,
    )

    random_grayscale = lambda x: transforms.RandomGrayscale(p=0.2)(x)

    return A.Compose(
        [color_jitter, A.Lambda(image=random_grayscale, name="random_grayscale", p=0.2)]
    )


class EmptyImage:
    """
    Class for generating blacked out images with a certain probability.

    Parameters
    ----------
    p : float
        The probability of an image being blacked out.

    Methods
    -------
    __call__(self, **images):
        Replace the input images with a blacked out image with a
        probability p, except the first image.

    """

    def __init__(self, p, **kwargs):
        self.p = p

    def add_targets(self, *args, **kwargs):
        pass

    def __call__(self, **images):
        transformed_images = {}
        for key, image in images.items():
            # don't apply transformation to the first image
            if key == "image":
                transformed_images[key] = image
                continue
            # with probability p the image is replaced with an array of zeroes (blacked out)
            if np.random.rand() < self.p:
                transformed_images[key] = np.zeros_like(image)
            else:
                transformed_images[key] = image
        return transformed_images
