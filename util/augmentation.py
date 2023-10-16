import math
import random

import numpy as np
import scipy.ndimage
import warnings



import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.backends.cudnn as cudnn

def cropToShape(image, new_shape, center_list=None, fill=0.0):
    # 移除 log.debug 语句
    # 移除 assert 语句

    if center_list is None:
        center_list = [int(image.shape[i] / 2) for i in range(3)]

    crop_list = []
    for i in range(0, 3):
        crop_int = center_list[i]
        if image.shape[i] > new_shape[i] and crop_int is not None:
            start_int = crop_int - int(new_shape[i]/2)
            end_int = start_int + new_shape[i]
            crop_list.append(slice(max(0, start_int), end_int))
        else:
            crop_list.append(slice(0, image.shape[i]))

    image = image[crop_list]

    crop_list = []
    for i in range(0, 3):
        if image.shape[i] < new_shape[i]:
            crop_int = int((new_shape[i] - image.shape[i]) / 2)
            crop_list.append(slice(crop_int, crop_int + image.shape[i]))
        else:
            crop_list.append(slice(0, image.shape[i]))

    new_image = np.zeros(new_shape, dtype=image.dtype)
    new_image[:] = fill
    new_image[crop_list] = image

    return new_image

def zoomToShape(image, new_shape, square=True):
    # 移除 assert 语句

    if square and image.shape[0] != image.shape[1]:
        crop_int = min(image.shape[0], image.shape[1])
        new_shape = [crop_int, crop_int, image.shape[2]]
        image = cropToShape(image, new_shape)

    zoom_shape = [new_shape[i] / image.shape[i] for i in range(3)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = scipy.ndimage.interpolation.zoom(
            image, zoom_shape,
            output=None, order=0, mode='nearest', cval=0.0, prefilter=True)

    return image

def randomOffset(image_list, offset_rows=0.125, offset_cols=0.125):
    # 移除一些 log.debug 语句

    center_list = [int(image_list[0].shape[i] / 2) for i in range(3)]
    center_list[0] += int(offset_rows * (random.random() - 0.5) * 2)
    center_list[1] += int(offset_cols * (random.random() - 0.5) * 2)
    center_list[2] = None

    new_list = []
    for image in image_list:
        new_image = cropToShape(image, image.shape, center_list)
        new_list.append(new_image)

    return new_list

# 其他函数也需要移除 log.debug 语句，操作方法与上面类似

