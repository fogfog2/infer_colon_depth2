# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as funct
import cv2
from glob import glob
from cv2 import imwrite, applyColorMap
import PIL.Image as pil

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image, image_grid
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor
from packnet_sfm.networks.layers.resnet.layers import disp_to_depth
from packnet_sfm.geometry.pose import Pose
from packnet_sfm.geometry.camera import Camera

import threading
import time

i= 0
pcd_list = []
view_pcd_list = []
axis_set_list = []
trajectory_list = []
fov_set_list = []
timer2 = 0

prev_axis = []

def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM inference of depth maps from images')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.ckpt)')
    parser.add_argument('--input', type=str, help='Input file or folder')
    parser.add_argument('--output', type=str, help='Output file or folder')
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    parser.add_argument('--save', type=str, choices=['npz', 'png'], default=None,
                        help='Save format (npz or png). Default is None (no depth map is saved).')
    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    assert (is_image(args.input) ) or \
           (not is_image(args.input) and not is_image(args.input)), \
        'Input and output must both be images or folders'
    return args

def path_loader(pts):

    min_z = pts[:,2].min()
    max_z = pts[:,2].max()
    step = 4
    res = (max_z - min_z)/step

    path = [[0,0,0]]
    for i in range(0,step):
        step_in = min_z + res*(i) 
        step_out = min_z + res*(i+1)
        idx = np.where(pts[:,2]< step_out)
        subptx = pts[idx]
        idx = np.where(subptx[:,2]>=step_in)
        test = subptx[idx]
        x = test[:,0].mean()
        y = test[:,1].mean()
        z = (step_in + step_out)/2.0
        path.append([x,y,z])
    return path

def path_loader_inv(pts):

    min_z = pts[:,2].min()
    max_z = pts[:,2].max()
    step = 10
    res = (max_z - min_z)/step

    path = [[0,0,0]]
    #path = []
    for i in range(9,step):
        step_in = max_z - res*(i+1) 
        step_out = max_z - res*(i)
        idx = np.where(pts[:,2]< step_out)
        subptx = pts[idx]
        idx = np.where(subptx[:,2]>=step_in)
        test = subptx[idx]
        x = test[:,0].mean()
        y = test[:,1].mean()
        z = (step_in + step_out)/2.0
        path.append([x,y,z])
    return path


def update_pose(PrevT, T):
    cam_to_world = np.dot( PrevT , T)
    xyzs = cam_to_world[:3, 3]
    return cam_to_world, xyzs

@torch.no_grad()
def infer_and_vis(input_file, prev_path, output_file, model_wrapper, image_shape, half, save ,  imagemode,prev_position):
    #print("infer_and_vis")
    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    if imagemode ==0:
        image = load_image(input_file).convert('RGB')
        prev_image = load_image(prev_path).convert('RGB')
    elif imagemode ==1:
        input_file = cv2.cvtColor(input_file, cv2.COLOR_BGR2RGB)
        image = pil.fromarray(input_file)

        prev_path = cv2.cvtColor(prev_path, cv2.COLOR_BGR2RGB)
        prev_image = pil.fromarray(prev_path)

    w, h = image.size

    # width_crop = 0
    # height_crop = 0
    # image = image.crop((width_crop,height_crop, w-width_crop, h-height_crop))

    # Resize and to tensor
    image = resize_image(image, image_shape)

    cv_image = np.array(image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    image = to_tensor(image).unsqueeze(0)
   
    # Load image    
    # Resize and to tensor
    prev_image = resize_image(prev_image, image_shape)
    prev_image = to_tensor(prev_image).unsqueeze(0)

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to('cuda:{}'.format(rank()), dtype=dtype)
        prev_image = prev_image.to('cuda:{}'.format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image)['inv_depths'][0]
    _, depth = disp_to_depth(pred_inv_depth, min_depth=0.1, max_depth=255.0)
   

    #camera intrinsic matrix
    K = np.array([[0.646, 0, 0.5, 0],
                [0, 0.6543, 0.5, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=np.float32)

    #camera intrinsic matrix scaling.float()
    K[0, :] *= 256//2**0
    K[1, :] *= 256//2**0

    K = K[0:3,0:3]
    K = to_tensor(K).to('cuda')

    tcw = Pose.identity(len(K)).to('cuda')
    cam = Camera(K=K, Tcw = tcw)

    world_points, view_world_points = cam.reconstruct2(depth, frame='w')

    world_points = world_points.view(1,3,256,256)    
    cam_points = world_points.view(3,-1).cpu().numpy()
    pts = np.transpose(cam_points)            
    pts = np.float64(pts)


    depth = depth *5.0
    depth_np = depth.permute(1,2,0).detach().cpu().numpy()*255    
    colored = cv2.applyColorMap(depth_np.astype(np.uint8), cv2.COLORMAP_JET)
    #colored = cv2.applyColorMap(depth_np.astype(np.uint8), cv2.COLORMAP_BONE)
    #colored = colored[:,:,::-1]
    axis = path_loader_inv(pts)
    target = axis[-1]
    #mean_depth = pts[:,2].mean()
    effect_depth = np.count_nonzero(pts[:,2]>0.05)
    
    max_depth = pts[:,2].max()
    idx = np.where(pts[:,2]==max_depth)
    cc = pts[idx][0]
    
    x_min = pts[:,0].min()
    x_max = pts[:,0].max()
    
    y_min = pts[:,1].min()    
    y_max = pts[:,1].max()

    norm_x = (cc[0] - x_min)/(x_max-x_min)
    norm_y = (cc[1] - y_min)/(y_max-y_min)

    width = image_shape[0]
    height = image_shape[1]

    # target_x = int(width- norm_x*width )
    # target_y = int(height- norm_y*height )

    target_x = int( norm_x*width )
    target_y = int( norm_y*height )
    
    
    #cv2.circle(cv_image, (target_x,target_y), 10, (0,255,255), 5)
    
    if effect_depth< 5000:
        current_color = (0,0,255)
        prev_position = [0,0,0]
    elif effect_depth >=5000 and effect_depth<10000:
        count = ((effect_depth-5000)/5000)*256
        current_color = (count,0,255-count)
        prev_position = [0,0,0]
    else:
        if prev_position[0]!=0 and prev_position[1]!=0:
            c_px = int((prev_position[0]+target_x)/2)
            c_py = int((prev_position[1]+target_y)/2)
        else:
            c_px = target_x
            c_py = target_y

        cv2.circle(cv_image, (c_px,c_py), 1+ (int)(max_depth*200), (0,255,255), 5)
        prev_position[0] = c_px
        prev_position[1] = c_py
        current_color = (255,0,0)
    
    prev_position[2] = max_depth
    cv2.rectangle(cv_image, (0,0) , (width, height), current_color, 5)
    
    added = cv2.hconcat([cv_image,colored])
    added = cv2.resize(added,(1024,512))
    cv2.imshow("tt",added)
    cv2.waitKey(1)
    return prev_position

    

def main(args):

    # Initialize horovod
    hvd_init()

    # Parse arguments
    config, state_dict = parse_test_file(args.checkpoint)

    # If no image shape is provided, use the checkpoint one
    image_shape = args.image_shape
    if image_shape is None:
        image_shape = config.datasets.augmentation.image_shape

    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)
    # Restore monodepth_model state
    model_wrapper.load_state_dict(state_dict)

    # change to half precision for evaluation if requested
    dtype = torch.float16 if args.half else None

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=dtype)

    # Set to eval mode
    model_wrapper.eval()  
    
    print("image_loader")
    prev_position = [0,0,0]
    if args.input is not None:
        if os.path.isdir(args.input):
            # If input file is a folder, search for image files
            files = []
            for ext in ['png', 'jpg']:
                files.extend(glob((os.path.join(args.input, '*.{}'.format(ext)))))
            files.sort()
            print0('Found {} files'.format(len(files)))

            for idx, image_path in enumerate(files):
                if idx ==0:
                    continue
                prev_path = files[idx-1]
                prev_position= infer_and_vis(image_path,prev_path, args.output, model_wrapper, image_shape, args.half, args.save, 0 ,prev_position)
                print("x :", prev_position[0]/image_shape[0], "y: ", prev_position[1]/image_shape[1], "z:", prev_position[2])

        elif os.path.isfile(args.input):
            cap=cv2.VideoCapture(args.input)
            prev_image = 0
            while(cap.isOpened()):
                ret, input_image = cap.read()
                if ret:
                    if prev_image is not 0:
                        prev_position= infer_and_vis(input_image,prev_image, args.output, model_wrapper, image_shape, args.half, args.save,  1,prev_position)
                        print("x :", prev_position[0]/image_shape[0], "y: ", prev_position[1]/image_shape[1], "z:", prev_position[2])
                    prev_image = input_image
        elif args.input=='cam':
            cap=cv2.VideoCapture(0)
            prev_image = 0
            while(cap.isOpened()):
                ret, input_image = cap.read()
                if ret:
                    if prev_image is not 0:
                        prev_position= infer_and_vis(input_image,prev_image, args.output, model_wrapper, image_shape, args.half, args.save,  1,prev_position)
                        print("x :", prev_position[0]/image_shape[0], "y: ", prev_position[1]/image_shape[1], "z:", prev_position[2])
                    prev_image = input_image

        else:
            # Otherwise, use it as is
            files = [args.input]
    
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
