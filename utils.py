import numpy as np
import nibabel as nib
import cv2 as cv


def load_nii(path):
    """Load image or nii file.

    Given a file path of nii file, load it and process it as images.

    Arguments:
        path {str} -- file path

    Returns:path
    	images {np.array} -- images range (0, 255) of shape (N, H, W)
		size {tuple} -- shape of images in format (N, H, W)
    """
    raw_obj = nib.load(path)
    affine = raw_obj.affine
    header = raw_obj.header
    raw_data = raw_obj.get_data().transpose(2, 1, 0).astype('float32')

    # Normalize to (0, 255)
    images = np.zeros_like(raw_data)
    for i, img in enumerate(raw_data):
        images[i, :, :] = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10) * 255
    size = images.shape[1:]  # (N, W, H)

    return images.astype('UInt8'), size, affine, header


def preprocess(images, denoise=False):
    ''' Preprocess images before feed into NN
    '''
    images = images.astype('float32')  # This is very important !!!!!!
    if denoise:
        for i, img in enumerate(images):
            deno = cv.fastNlMeansDenoising(img)
            images[i, :, :] = (deno - np.mean(deno)) / np.std(deno)
    else:
        for i, img in enumerate(images):
            images[i, :, :] = (img - np.mean(img)) / np.std(img)

    return images


def image_resize(image, width=None, height=None, inter=cv.INTER_NEAREST):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(np.ceil(w * r)), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(np.ceil(h * r)))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return np.flip(resized, 1)


############################################
def postprocess_numpy(prob_map):
    ''' Convert prob map from NN to mask 

    Arguments:
        prob_map {np.ndarray} -- Probability map generated from NN, in shape (N, C, W, H)

    Returns:
        mask {np.array} -- Postprocessed mask for result visualization
    '''
    preds = prob_map.squeeze(0).squeeze(0)
    probs = 1 / (1 + np.exp(-preds))
    probs[probs > 0.5] = 255
    probs[probs != 255] = 0

    return probs


def preprocess_numpy(images, denoise=False):
    ''' Preprocess images before feed into NN
    '''
    images = images.astype('float32')  # This is very important !!!!!!
    if denoise:
        for i, img in enumerate(images):
            deno = cv.fastNlMeansDenoising(img)
            images[i, :, :] = (deno - np.mean(deno)) / np.std(deno)
    else:
        for i, img in enumerate(images):
            images[i, :, :] = (img - np.mean(img)) / np.std(img)

    return images


