#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fine-tune an ASiT model on a new dataset.
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
import datetime
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision

import numpy as np

import dataloader
from data_transformations import DataAugmentation
import main_ASiT
import vision_transformer as vits
from vision_transformer import CLSHead, RECHead

# Assume utils and vision_transformer are modules provided in your project

def get_args_parser():
    # parser = argparse.ArgumentParser('ASiT Fine-Tuning', add_help=False)
    # add necessary arguments
    # parser.add_argument('--data_path', default='/path/to/data', type=str, help='Path to the dataset')
    # parser.add_argument('--data_json', default='/path/to/dataset.json', type=str, help='Path to the dataset JSON file')
    # parser.add_argument('--output_dir', default='./fine_tune_checkpoints', type=str, help='Directory for output')
    # parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to fine-tune')
    # parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate for fine-tuning')

    parser = argparse.ArgumentParser('ASiT Fine-Tuning', add_help=False)

    parser.add_argument('--ckpt_path', default='/path/to/pretrained/model.pth', type=str, help='Path to the pretrained model checkpoint')


    # Model parameters
    parser.add_argument('--arch', default='vit_base', type=str, 
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help="architecture Name")
    parser.add_argument('--patch_size', default=16, type=int, help="Patch size in pixels")
 
    # Reconstruction parameters
    parser.add_argument('--recons_blocks', default='6-8-10-12', type=str, help="""Reconstruct the input back from the 
                        given blocks, empty string means no reconstruction will be applied. (Default: '6-8-10-12') """)
    parser.add_argument('--drop_perc', type=float, default=0.5, help='Drop X percentage of the input image')
    parser.add_argument('--drop_replace', type=float, default=0.3, help='Drop X percentage of the input image')

    parser.add_argument('--drop_align', type=int, default=1, help='Align drop with patches')
    parser.add_argument('--drop_type', type=str, default='zeros', help='Drop Type.')
    parser.add_argument('--drop_only', type=int, default=1, help='Align drop with patches')
    
    parser.add_argument('--fromINet', default=1, type=int, help="Start the training from ImageNet pre-trained weights")
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sampling Rate')
  

    parser.add_argument('--out_dim', default=8192, type=int, help="Dimensionality of the head output.")
    # parser.add_argument('--norm_last_layer', default=False, type=utils.bool_flag, 
    #                     help="Whether or not to weight normalize the last layer")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="Base EMA")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float, help="Initial value for the teacher temperature.")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="Final value (after linear warmup).")
    parser.add_argument('--warmup_teacher_temp_epochs', default=5, type=int, help='Number of warmup epochs for the teacher temperature.')

    # Training/Optimization parameters
    # parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="Use half precision for training.")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="Initial value of the weight decay.")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="Final value of the weight decay.")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="Maximal parameter gradient norm if using gradient clipping.")
    parser.add_argument('--batch_size', default=20, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="Number of epochs during which we keep the output layer fixed.")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate.""")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the end of optimization. """)
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    parser.add_argument("--data_path", type=str, 
                        default='./KAGGLE/AUDIO', 
                        help="dataset path")
    parser.add_argument("--data_json", type=str, default='./dataset.json', help="training data json")
    parser.add_argument("--num_frames", default=592,type=int, help="the input length in frames")
    parser.add_argument("--num_mel_bins", type=int, default=128, help="number of input mel bins")
    parser.add_argument("--data_mean", type=float, default=-4.2677393, help="the dataset mean, used for input normalization")
    parser.add_argument("--data_std", type=float, default=4.5689974, help="the dataset std, used for input normalizations")
    
    parser.add_argument('--num_crops', type=int, default=2, help='number of seconds to crop during augmentation')
    parser.add_argument('--secs_per_crop', type=int, default=6, help='number of seconds to crop during augmentation')
    
    parser.add_argument("--num_frames_local", default=192,type=int, help="the input length in frames")
    parser.add_argument('--num_crops_local', type=int, default=4, help='number of seconds to crop during augmentation')
    parser.add_argument('--secs_per_crop_local', type=int, default=2, help='number of seconds to crop during augmentation')
    
    parser.add_argument('--output_dir', default="checkpoints/vit_base/AUDIOSet", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def load_pretrained_model(student, path):
    checkpoint = torch.load(path, map_location='cpu')
    student.load_state_dict(checkpoint['student'], strict=False)
    print("Pre-trained model loaded from", path)

def freeze_layers(model):
    for name, param in model.named_parameters():
        if 'head' not in name:  # Freeze layers that are not part of the classification head
            param.requires_grad = False

def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    # sampler = data_loader.sampler
    # sampler.set_epoch(epoch)  # Properly shuffle for distributed training

    for i, (inputs, labels) in enumerate(data_loader):
        if inputs is None:  # Skip batches with no valid data
            continue
        
        for input in inputs:
            print(len(input))
            # print(input.shape)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:  # Print loss every 10 batches
            print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the pre-trained model
    model = vits.__dict__['vit_base'](num_classes=2)  # Adjust number of classes
    load_pretrained_model(model, args.ckpt_path)
    freeze_layers(model)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Data
    transform = DataAugmentation(args)
    dataset = dataloader.AudioDataset(args.data_json, args.data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=main_ASiT.collate_batch(0.0, 1)
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    for epoch in range(args.epochs):
        train_one_epoch(model, data_loader, optimizer, device, epoch)

    # Save the fine-tuned model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'fine_tuned.pth'))
    print("Fine-tuning completed and model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASiT Fine-Tuning', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
