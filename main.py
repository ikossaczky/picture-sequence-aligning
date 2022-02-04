import os
from concurrent.futures import ProcessPoolExecutor

from copy import deepcopy
import cv2
import numpy as np

WARP_MODE = cv2.MOTION_AFFINE
NUMBER_OF_ITERATIONS = 5000
TERMINATION_STEPS = 1e-10


def image_loading_generator(folder, preprocessor=None):
    for image in sorted(os.listdir(folder)):
        im = cv2.imread(os.path.join(folder, image))
        if preprocessor is not None:
            im = preprocessor(im)
        yield im


def simple_preprocessor(resize_factor: float = 0.1, cutout=None):
    def _preprocessor(im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, tuple(reversed([int(x * resize_factor) for x in im.shape])), interpolation=cv2.INTER_AREA)
        if cutout is not None:
            im = im[cutout[0][0]:cutout[0][1], cutout[0][0]:cutout[0][1]]
        return im
    return _preprocessor


def _get_init_transformation():
    if WARP_MODE == cv2.MOTION_HOMOGRAPHY:
        return np.eye(3, 3, dtype=np.float32)
    else:
        return np.eye(2, 3, dtype=np.float32)


def compute_transformation(image1, image2):
    assert image1.shape == image2.shape

    transf = _get_init_transformation()
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, NUMBER_OF_ITERATIONS, TERMINATION_STEPS)

    _, transf = cv2.findTransformECC(image1, image2, transf, WARP_MODE, criteria, inputMask=None,
                                          gaussFiltSize=1)

    return transf


def get_transformation_multiproc(image_iter1, image_iter2):
    iter1, iter2 = iter(image_iter1), iter(image_iter2)
    _ = next(iter2)
    with ProcessPoolExecutor() as executor:
        return executor.map(compute_transformation, iter1, iter2)


def rescale_transformation(transf, factor):
    assert transf.shape == (2, 3)
    rescaled_transf = deepcopy(transf)
    rescaled_transf[:, -1:] = factor * rescaled_transf[:, -1:]
    return rescaled_transf


def accumulate_transformations(transf_list: list):
    raise NotImplementedError()


def transform(image, transf):
    shp = image.shape
    if WARP_MODE == cv2.MOTION_HOMOGRAPHY:
        image_aligned = cv2.warpPerspective(image, transf, (shp[1], shp[0]),
                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        image_aligned = cv2.warpAffine(image, transf, (shp[1], shp[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return image_aligned


if __name__ == "__main__":
    image_directory = None

    preproc = simple_preprocessor()
    loader1 = image_loading_generator(image_directory, preproc)
    loader2 = image_loading_generator(image_directory, preproc)
    next(loader2)

    transformations = get_transformation_multiproc(loader1, loader2)
    transformations = accumulate_transformations(list(transformations))

