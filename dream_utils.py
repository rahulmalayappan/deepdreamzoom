import torch
from torchvision import transforms
import cv2
import numpy as np
import os

import image_utils
import time

def gradient_ascent_image(img_tensor, loss_fn, num_iterations, lr, max_loss=None):
    assert num_iterations >= 1 and lr >= 0.
    
    img_tensor.requires_grad = True
    if img_tensor.grad is not None:
        img_tensor.grad.zero_()
        
    for i in range(num_iterations):
        loss = loss_fn(img_tensor)
        
        if max_loss is not None and loss.item() > max_loss:
            return loss.detach().item()
        
        loss.backward()
        
        with torch.no_grad():
            delta = img_tensor.grad
            delta_scale = max(torch.abs(delta).mean().item(), 1e-6) # Try other forms of normalization
            
            img_tensor += delta * lr / delta_scale
            
        img_tensor.grad.zero_()
        
    return loss.detach().item()


def deepdream_image(img_tensor, loss_fn, num_iterations, lr, octave_scale=1.4, num_octaves=3):
    h = img_tensor.shape[2]
    w = img_tensor.shape[3]
    
    # Smallest to largest
    shapes = [(int(h / (octave_scale ** i)), int(w / (octave_scale ** i))) for i in range(num_octaves)][::-1]
    orig_imgs = [transforms.Resize(shape)(img_tensor) for shape in shapes]
    
    ascended_diff = None
    for i in range(len(orig_imgs)):
        ascended = orig_imgs[i].detach().clone()
        if ascended_diff is not None:
            ascended += transforms.Resize(shapes[i])(ascended_diff)
            
        loss_value = gradient_ascent_image(ascended, loss_fn, num_iterations, lr)
        ascended_diff = (ascended - orig_imgs[i]).detach()
        
    return ascended

def deepdream_zoom(base_img, zoom_config, loss_fn):
    # Parse config
    lr = zoom_config['lr']
    num_iters = zoom_config['num_iters']
    num_octaves = zoom_config['num_octaves']

    zoom_amount = zoom_config['zoom_amount']
    rotation_degrees = zoom_config['rotation_degrees']

    num_ease_in_frames = zoom_config['num_ease_in_frames']
    ease_in_power = zoom_config['ease_in_power']
    
    num_frames = zoom_config['num_frames']
    output_dir = zoom_config['output_dir']

    frame = base_img

    # Pre zoom
    h = frame.shape[0]
    w = frame.shape[1]
    rotation_radians = rotation_degrees * np.pi / 180.

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    start_time = time.time()
    for i in range(num_frames):
        if num_ease_in_frames == 0:
            ease_in_scale = 1.
        else:
            ease_in_factor = float(i) / num_ease_in_frames
            ease_in_scale = min(1., ease_in_factor ** ease_in_power)

        A, b = image_utils.compose_affine(
            *image_utils.gen_translation(w / 2, h / 2), # Translate image center to origin
            *image_utils.compose_affine(
                *image_utils.gen_scale(1. + zoom_amount * ease_in_scale), # Then zoom a bit
                *image_utils.compose_affine(
                    *image_utils.gen_rotation(rotation_radians * ease_in_scale), # Then rotate a bit
                    *image_utils.gen_translation(-w / 2, -h / 2) # Then put back the center
                )
            )
        )

        frame = image_utils.tensor_to_img(
            deepdream_image(
                image_utils.img_to_tensor(frame).to("cuda"),
                loss_fn, num_iters, lr * ease_in_scale, num_octaves=num_octaves
            ).detach().to("cpu"))

        frame = cv2.warpAffine(frame, np.column_stack((A, b)), (frame.shape[1], frame.shape[0]))

        image_utils.write_img(frame, os.path.join(output_dir, "{}.jpg".format(i)))
        
        if i % 10 == 0:
            avg_time = (time.time() - start_time) / (i + 1)
            print("{}, average time: {:.4f}".format(i, avg_time))
