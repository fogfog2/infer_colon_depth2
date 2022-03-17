# Copyright 2020 Toyota Research Institute.  All rights reserved.

import random

from packnet_sfm.geometry.pose import Pose
from packnet_sfm.models.base_model import BaseModel
from packnet_sfm.models.model_utils import flip_batch_input, flip_output, upsample_output
from packnet_sfm.utils.misc import filter_dict


import torch
import torch.nn as nn
class SfmModel(BaseModel):
    """
    Model class encapsulating a pose and depth networks.

    Parameters
    ----------
    depth_net : nn.Module
        Depth network to be used
    pose_net : nn.Module
        Pose network to be used
    rotation_mode : str
        Rotation mode for the pose network
    flip_lr_prob : float
        Probability of flipping when using the depth network
    upsample_depth_maps : bool
        True if depth map scales are upsampled to highest resolution
    kwargs : dict
        Extra parameters
    """
    def __init__(self, depth_net=None, depth_net_feedback=None, pose_net=None,
                 rotation_mode='euler', flip_lr_prob=0.0,
                 upsample_depth_maps=False, **kwargs):
        super().__init__()
        self.depth_net = depth_net
        self.depth_net_feedback = depth_net_feedback
        self.pose_net = pose_net
        self.rotation_mode = rotation_mode
        self.flip_lr_prob = flip_lr_prob
        self.upsample_depth_maps = upsample_depth_maps

        self._network_requirements = [
            'depth_net',
            'depth_net_feedback',
            'pose_net',
        ]

    def add_depth_net(self, depth_net):
        """Add a depth network to the model"""
        self.depth_net = depth_net

    def add_depth_net_feedback(self, depth_net_feedback):
        """Add a depth network to the model"""
        self.depth_net_feedback = depth_net_feedback

    def add_pose_net(self, pose_net):
        """Add a pose network to the model"""
        self.pose_net = pose_net

    def depth_net_flipping(self, batch, flip):
        """
        Runs depth net with the option of flipping

        Parameters
        ----------
        batch : dict
            Input batch
        flip : bool
            True if the flip is happening

        Returns
        -------
        output : dict
            Dictionary with depth network output (e.g. 'inv_depths' and 'uncertainty')
        """
        # Which keys are being passed to the depth network
        temp = {key: batch[key] for key in filter_dict(batch, ['rgb_context'])}
        #batch['rgb_context'] = batch['rgb_context'][0]
        #temp['rgb_context'][0]
        if flip:
            # Run depth network with flipped inputs
            output = self.depth_net(flip_batch_input(temp)['rgb_context'] [0])
            # Flip output back if training
            output = flip_output(output)
        else:
            # Run depth network
            output = self.depth_net(temp['rgb_context'][0])
        return output

    def depth_net_feedback_flipping(self, batch, disp, flip):
        """
        Runs depth net with the option of flipping

        Parameters
        ----------
        batch : dict
            Input batch
        flip : bool
            True if the flip is happening

        Returns
        -------
        output : dict
            Dictionary with depth network output (e.g. 'inv_depths' and 'uncertainty')
        """
        # Which keys are being passed to the depth network
        
        batch_input = {key: batch[key] for key in filter_dict(batch, self._input_keys)}

        if self.training:
            batch_input['rgb']  = torch.cat([ batch_input['rgb'] , disp['inv_depths'][0] ] , 1)
        else:
            batch_input['rgb']  = torch.cat([ batch_input['rgb'] , disp['inv_depths'] ] , 1)

        if flip:
            # Run depth network with flipped inputs
            
            output = self.depth_net_feedback(**flip_batch_input(batch_input))
            # Flip output back if training
            output = flip_output(output)
        else:
            # Run depth network
            output = self.depth_net_feedback(**batch_input)
        return output

    def compute_depth_net(self, batch, force_flip=False):
        """Computes inverse depth maps from single images"""
        # Randomly flip and estimate inverse depth maps
        output = self.depth_net_flipping(batch, force_flip)
        # If upsampling depth maps at training time
        if self.training and self.upsample_depth_maps:
            output = upsample_output(output, mode='nearest', align_corners=None)
        # Return inverse depth maps
        return output

    def compute_depth_net_feedback(self, batch, depth_output, force_flip=False):
        """Computes inverse depth maps from single images"""
        # Randomly flip and estimate inverse depth maps
        output = self.depth_net_feedback_flipping(batch, depth_output, force_flip)
        # If upsampling depth maps at training time
        if self.training and self.upsample_depth_maps:
            output = upsample_output(output, mode='nearest', align_corners=None)
        # Return inverse depth maps
        return output

    def compute_pose_net(self, image, contexts):
        """Compute poses from image and a sequence of context images"""
        pose_vec = self.pose_net(image, contexts)
        return [Pose.from_vec(pose_vec[:, i], self.rotation_mode)
                for i in range(pose_vec.shape[1])]

    def forward(self, batch, return_logs=False, force_flip=False):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        force_flip : bool
            If true, force batch flipping for inverse depth calculation

        Returns
        -------
        output : dict
            Dictionary containing the output of depth and pose networks
        """


        flag_flip_lr = random.random() < self.flip_lr_prob if self.training else force_flip

        # Generate inverse depth predictions
        depth_output = self.compute_depth_net(batch, force_flip=flag_flip_lr)

        depth_feedback_output = self.compute_depth_net_feedback(batch, depth_output, force_flip=flag_flip_lr)

        # Generate pose predictions if available
        pose_output = None
        if 'rgb_context' in batch and self.pose_net is not None:
            pose_output = self.compute_pose_net(
                batch['rgb'], batch['rgb_context'])
        # Return output dictionary
        return {
            'poses': pose_output,
            'prev_inv_depths' : depth_output['inv_depths'],
            'inv_depths' : depth_feedback_output['inv_depths'],
        }
