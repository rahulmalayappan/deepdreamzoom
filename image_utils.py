import torch
from torch import nn
import cv2
import numpy as np

def np_to_torch(x):
    return np.swapaxes(np.swapaxes(x, 0, 2), 1, 2)

def torch_to_np(x):
    return np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)

def load_img(filename):
    return cv2.imread(filename)[:, :, ::-1] / 255.

def write_img(img, filename):
    cv2.imwrite(filename, (img[:, :, ::-1] * 255).astype(np.uint8))
    
def img_to_tensor(img):
    return torch.Tensor(np_to_torch(img)).unsqueeze(0)

def tensor_to_img(img_tensor):
    return np.maximum(np.minimum(torch_to_np(np.array(img_tensor[0])), 1.), 0.)

# Use OpenCV Affine transformations since torch transforms
# don't allow you to resample after composing

# Affine transformations return tuple (A, b)
def compose_affine(A0, b0, A1, b1):
    return np.dot(A0, A1), np.dot(A0, b1) + b0

def gen_translation(x, y):
    return np.eye(2), np.array([x, y])

def gen_rotation(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ]), np.array([0., 0.])

def gen_scale(s):
    return np.eye(2) * s, np.array([0., 0.])