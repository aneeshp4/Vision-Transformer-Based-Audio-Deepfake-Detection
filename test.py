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
import main_ASIT
import vision_transformer as vits
from vision_transformer import CLSHead, RECHead

# Assume utils and vision_transformer are modules provided in your project

def get_args_parser():
    parser = argparse.ArgumentParser('ASiT Fine-Tuning', add_help=False)
    # add necessary arguments
    parser.add_argument('--data_path', default='/path/to/data', type=str, help='Path to the dataset')
    parser.add_argument('--data_json', default='/path/to/dataset.json', type=str, help='Path to the dataset JSON file')
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

def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    sampler = data_loader.sampler
    sampler.set_epoch(epoch)  # Properly shuffle for distributed training

    for i, (inputs, labels) in enumerate(data_loader):
        if inputs is None:  # Skip batches with no valid data
            continue
        
        inputs, labels = inputs.to(device), labels.to(device)
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
    dataset = dataloader.AudioDataset(args.data_train, args.data_path, sample_rate=args.sample_rate, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=main_ASIT.collate_batch(args.drop_replace, args.drop_align)
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
