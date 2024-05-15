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
import utils
import vision_transformer as vits
from vision_transformer import CLSHead, RECHead

# Assume utils and vision_transformer are modules provided in your project

def get_args_parser():
    parser = argparse.ArgumentParser('ASiT Fine-Tuning', add_help=False)
    # add necessary arguments
    parser.add_argument('--data_path', default='/path/to/data', type=str, help='Path to the dataset')
    parser.add_argument('--ckpt_path', default='/path/to/pretrained/model.pth', type=str, help='Path to the pretrained model checkpoint')
    parser.add_argument('--output_dir', default='./fine_tune_checkpoints', type=str, help='Directory for output')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to fine-tune')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate for fine-tuning')
    return parser

def load_pretrained_model(student, path):
    checkpoint = torch.load(path, map_location='cpu')
    student.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("Pre-trained model loaded from", path)

def freeze_layers(model):
    for name, param in model.named_parameters():
        if 'head' not in name:  # Freeze layers that are not part of the classification head
            param.requires_grad = False

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the pre-trained model
    student = vits.__dict__['vit_base'](num_classes=1000)  # Adjust number of classes
    load_pretrained_model(student, args.ckpt_path)
    freeze_layers(student)
    student.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr)

    # Data
    train_dataset = dataloader.YourDataset(args.data_path, transform=DataAugmentation())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    # Training loop
    student.train()
    for epoch in range(args.epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Save the fine-tuned model
    torch.save(student.state_dict(), os.path.join(args.output_dir, 'fine_tuned.pth'))
    print("Fine-tuning completed and model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASiT Fine-Tuning', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
