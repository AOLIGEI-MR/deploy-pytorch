import numpy as np
from PIL import Image
import os
import nibabel as nib
import cv2 as cv
from PyQt5 import QtGui
import datetime
import wmi
import json
import base64
import pydicom

def load_image(path):
    """Load image or nii file.

    Given a file path of image, load it and process it as images.

    Arguments:
        path {str} -- file path

    Returns:path
    	image {np.array} -- images range (0, 255) of shape (N, H, W)
		size {tuple} -- shape of images in format (N, H, W)
    """

    image = np.flipud(np.array(Image.open(path)))  # Here the flip is for image file only
    image = np.expand_dims(image, axis=0)
    size = image.shape

    return image.astype('UInt8'), size


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



def load_dcm(dcm_files):
    # Obtain number of dicoms to be processed
    dcm_num = len(dcm_files)
    # Container for dicom content
    dcm_all_info = []

    for file in dcm_files:
        dcm = pydicom.dcmread(file)
        sliceLoc = float(dcm.SliceLocation)
        name = os.path.basename(file)

        # Collect slice location info for sorting
        dcm_all_info.append((sliceLoc, dcm, name)) # (slice location, dicom content, file name)

    dim = dcm_all_info[0][1].pixel_array.shape
    # Case for single dicom
    if dcm_num == 1:
        array = np.int16(dcm_all_info[0][1].pixel_array)[np.newaxis,...]
        data = array
        
    # Case for multiple dicoms
    elif dcm_num > 1:
        array3D = None
        
        # Sort all dicom files according to their slice location
        dcm_all_info.sort(reverse=True)

        for item in dcm_all_info:
            dcm = item[1]
            imageData = dcm.pixel_array
            assert imageData.shape == dim, 'Images are not in the same dimension'

            imageData16 = np.int16(imageData)[np.newaxis,...]

            if array3D is None:
                array3D = imageData16
            else:
                array3D = np.concatenate((array3D, imageData16), axis=0)

        data = array3D
    
    images = np.zeros_like(data)
    for i, img in enumerate(data):
        images[i, :, :] = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10) * 255
        
    return dcm_all_info, images, dim

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


def overlayPlot(image, mask, style='overlap'):
    '''Plot lesion mask on original image

    Arguments:
        image {np.array} -- original MR images of shape (H, W)
        mask {np.array} -- generated mask of shape (H, W)

    Keyword Arguments:
        style {str} -- select plot style (default: {'overlap'})

    Returns:
        result {np.array} -- Overlay result of shape (H, W, 3)
    '''
    image_3c = np.tile(image[..., None], 3).astype('float32')
    mask_3c = np.tile(mask[..., None], 3).astype('float32')

    if style == 'overlap':
        mask_3c[:, :, 0] = mask_3c[:, :, 0] * 0.2
        mask_3c[:, :, 2] = mask_3c[:, :, 1] * 0.2

        result = cv.addWeighted(image_3c, 1, mask_3c, 0.4, 0)
        result[result > 255] = 255

    if style == 'outline':
        mask = np.round(mask)
        yy, xx = np.nonzero(mask)
        for y, x in zip(yy, xx):
            if 0.0 < np.mean(mask[max(0, y - 1): y + 2, max(0, x - 1): x + 2]) < 255.0:
                image_3c[max(0, y): y + 1, max(0, x): x + 1] = [0, 255, 0]

        result = image_3c

    return result.astype('UInt8')


def toSquare(image):
    '''Pad image to squared shape'''
    image = Image.fromarray(image)
    x, y = image.size
    size = max(x, y)
    if image.mode == 'L':
        new_im = Image.new('P', (size, size), 'black')
    else:
        new_im = Image.new('RGB', (size, size), 'black')
    new_im.paste(image, (int((size - x) / 2), int((size - y) / 2)))

    return np.array(new_im)


gray_color_table = [QtGui.qRgb(i, i, i) for i in range(256)]


def toQImage(im, copy=False):
    h, w = im.shape[:2]
    if h != w:
        im = toSquare(im)
    im = np.flip(im, 1)
    im = np.require(im, np.uint8, 'C')
    if im is None:
        return QtGui.QImage()

    if im.dtype == np.uint8:
        if len(im.shape) == 2:
            qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_Indexed8)
            qim.setColorTable(gray_color_table)
            return qim.copy() if copy else qim

        elif len(im.shape) == 3:
            if im.shape[2] == 3:
                qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888);
                return qim.copy() if copy else qim
            elif im.shape[2] == 4:
                qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32);
                return qim.copy() if copy else qim


def QImageToCvMat(incomingImage):
    '''  Converts a QImage into an opencv MAT format  '''

    incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format.Format_RGB32)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(height * width * 4)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))  # can be in BGR format
    return arr[..., :3]


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


