import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
 
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F 
import torchvision

import dataloader
from data_transformations import GMML_replace_list, DataAugmentation


import utils
import vision_transformer as vits
from vision_transformer import CLSHead, RECHead

def get_args_parser():
    parser = argparse.ArgumentParser('ASiT', add_help=False)

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
    parser.add_argument('--norm_last_layer', default=False, type=utils.bool_flag, 
                        help="Whether or not to weight normalize the last layer")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="Base EMA")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float, help="Initial value for the teacher temperature.")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="Final value (after linear warmup).")
    parser.add_argument('--warmup_teacher_temp_epochs', default=5, type=int, help='Number of warmup epochs for the teacher temperature.')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="Use half precision for training.")
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

    # Dataset
    parser.add_argument("--data_path", type=str, 
                        default='AUDIO/audioset_tagging_cnn/dataset_root/audios', 
                        help="dataset path")
    parser.add_argument("--data-train", type=str, default='AUDIO_Files/clean_data_combined.json', help="training data json")
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


class collate_batch(object): # replace from other images
    def __init__(self, drop_replace=0., drop_align=1):
        self.drop_replace = drop_replace
        self.drop_align = drop_align

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)

        if self.drop_replace > 0:
            batch[0][1][0], batch[0][2][0] = GMML_replace_list(batch[0][0][0], batch[0][1][0], batch[0][2][0],
                                                                            max_replace=self.drop_replace, align=self.drop_align)
            batch[0][1][1], batch[0][2][1] = GMML_replace_list(batch[0][0][1], batch[0][1][1], batch[0][2][1],
                                                                            max_replace=self.drop_replace, align=self.drop_align)

        return batch



def get_shape(fstride, tstride, patch_size, input_fdim=128, input_tdim=1024):
    test_input = torch.randn(1, 1, input_fdim, input_tdim)
    test_proj = nn.Conv2d(1, 768, kernel_size=(patch_size, patch_size), stride=(fstride, tstride))
    test_out = test_proj(test_input)
    f_dim = test_out.shape[2]
    t_dim = test_out.shape[3]
    return f_dim, t_dim

def train_ASiT(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
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
        collate_fn=collate_batch(args.drop_replace, args.drop_align)
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    student = vits.__dict__[args.arch](audio_size=[args.num_frames, args.num_mel_bins], in_chans=1, drop_path_rate=args.drop_path_rate)
    teacher = vits.__dict__[args.arch](audio_size=[args.num_frames, args.num_mel_bins], in_chans=1)
    embed_dim = student.embed_dim


    # self-supervised learning projection heads
    cls_head_s = CLSHead(embed_dim, args.out_dim, norm_last_layer=args.norm_last_layer)
    img_recons_s = RECHead(embed_dim, [args.num_frames, args.num_mel_bins], patch_size=args.patch_size, in_chans=1)
    student = FullPipeline(student, cls_head_s, img_recons_s)

    cls_head_t = CLSHead(embed_dim, args.out_dim)
    img_recons_t = RECHead(embed_dim, [args.num_frames, args.num_mel_bins], patch_size=args.patch_size, in_chans=1)
    teacher = FullPipeline(teacher, cls_head_t, img_recons_t)
    
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    
    # Load backbone weights from ImageNet pretrained weights
    if args.fromINet == 1:
        checkpoint = torch.load("checkpoints/" + args.arch + "/INet_Pretrained/checkpoint.pth", map_location="cpu")
        
        pos_embed_checkpoint = checkpoint['state_dict']['pos_embed']
        orig_size = [224//16, 224//16]
        new_size = [args.num_frames//16, args.num_mel_bins//16]

        extra_tokens = pos_embed_checkpoint[:, :1]
        pos_tokens = pos_embed_checkpoint[:, 1:]
        pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embed_dim).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size[0], new_size[1]), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint['state_dict']['pos_embed'] = new_pos_embed

        checkpoint['state_dict']['patch_embed.proj.weight'] = torch.sum(checkpoint['state_dict']['patch_embed.proj.weight'], dim=1).unsqueeze(1)
        
        msg = student.module.backbone.load_state_dict(checkpoint['state_dict'], strict=False)
        print(msg)
    
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
        
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # Preparing loss
    CLS_PATCH_loss = CLS_PATCH_Loss(args.out_dim, 2 + args.num_crops_local, 
        args.warmup_teacher_temp, args.teacher_temp,
        args.warmup_teacher_temp_epochs, args.epochs).cuda()
 
    # Preparing optimizer
    params_groups = utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)  

    # for mixed precision training
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # Init schedulers
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256.,
        args.min_lr, args.epochs, len(data_loader),  warmup_epochs=args.warmup_epochs)
    
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, len(data_loader))
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print("Loss, optimizer and schedulers ready.")

    # Resume training if checkpoint exist!
    start_epoch = 0
    
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore, student=student, teacher=teacher,
        optimizer=optimizer, fp16_scaler=fp16_scaler, CLS_PATCH_loss=CLS_PATCH_loss)
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, CLS_PATCH_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs  ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'CLS_PATCH_loss': CLS_PATCH_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, CLS_PATCH_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    
    save_recon = os.path.join(args.output_dir, 'reconstruction_samples')
    Path(save_recon).mkdir(parents=True, exist_ok=True)
    bz = args.batch_size
    plot_ = True


    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, ((clean_crops, corrupted_crops, masks_crops), _) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        clean_crops = [im.cuda(non_blocking=True) for im in clean_crops]
        corrupted_crops = [im.cuda(non_blocking=True) for im in corrupted_crops]
        masks_crops = [im.cuda(non_blocking=True) for im in masks_crops]

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output, t_data, _ = teacher(clean_crops[:2], recons=False)
 
            rloss = 0.0
            student_output, s_data, s_recons = student(corrupted_crops)
            if args.num_crops_local > 0:
                student_output_l, _, _ = student(clean_crops[2:], local=True)
                student_output = torch.cat ( (student_output, student_output_l), dim=0)
 
 
        
            #-------------------------------------------------
            rloss = 0.
            recloss = F.l1_loss(s_recons, torch.cat(clean_crops[0:2]), reduction='none')
            rloss = recloss[torch.cat(masks_crops[0:2])==1].mean() if (args.drop_only == 1) else recloss.mean()

            if plot_==True and utils.is_main_process():
                plot_ = False
                #validating: check the reconstructed images
                print_out = save_recon + '/epoch_' + str(epoch).zfill(5)  + '.jpg'
                imagesToPrint = torch.cat([clean_crops[0][0: min(5, bz)].permute(0, 1, 3, 2).cpu(),  corrupted_crops[0][0: min(5, bz)].permute(0, 1, 3, 2).cpu(),
                               s_recons[0: min(5, bz)].permute(0, 1, 3, 2).cpu(), masks_crops[0][0: min(5, bz)].permute(0, 1, 3, 2).cpu(),
                               clean_crops[1][0: min(5, bz)].permute(0, 1, 3, 2).cpu(),  corrupted_crops[1][0: min(5, bz)].permute(0, 1, 3, 2).cpu(),
                               s_recons[bz: bz+min(5, bz)].permute(0, 1, 3, 2).cpu(), masks_crops[1][0: min(5, bz)].permute(0, 1, 3, 2).cpu()], dim=0)
                torchvision.utils.save_image(imagesToPrint, print_out, nrow=min(5, bz), normalize=True, range=(-1, 1))
 
            
            closs, dloss = CLS_PATCH_loss(student_output, teacher_output, s_data, t_data, epoch)

        loss = (closs + dloss)/2 + rloss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(closs=closs.item() if hasattr(closs, 'item') else 0.)
        metric_logger.update(dloss=dloss.item() if hasattr(dloss, 'item') else 0.)
        metric_logger.update(rloss=rloss.item() if hasattr(rloss, 'item') else 0.)
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

 

class CLS_PATCH_Loss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, out_dim))
 
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output,  s_output, t_output, epoch):
 
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        
        s_output_ = s_output / self.student_temp
        s_output_ = s_output_.chunk(2)

        # teacher centering and sharpening
        t_output_ = F.softmax((t_output - self.center2) / temp, dim=-1)
        t_output_ = t_output_.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        
        total_loss2 = 0
        n_loss_terms2 = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    dloss = torch.sum(-t_output_[iq] * F.log_softmax(s_output_[v], dim=-1), dim=-1)
                    total_loss2 += dloss.mean()
                    n_loss_terms2 += 1
                    continue
                    
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        total_loss2 /= n_loss_terms2
        self.update_center(teacher_output, t_output, epoch)
        return total_loss, total_loss2

    @torch.no_grad()
    def update_center(self, teacher_output, teacher_output_data, epoch):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        
        
        B, N, _ = teacher_output_data.size()
        batch_center2 = torch.sum(torch.sum(teacher_output_data, dim=0, keepdim=True), dim=1, keepdim=True)
        dist.all_reduce(batch_center2)
        batch_center2 = batch_center2 / (B * N * dist.get_world_size())

        self.center2 = self.center2 * self.center_momentum + batch_center2 * (1 - self.center_momentum)



class FullPipeline(nn.Module):
    def __init__(self, backbone, head, head_recons):
        super(FullPipeline, self).__init__()

        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.head_recons = head_recons

    def forward(self, x, recons=True, local=False):
        _out = self.backbone(torch.cat(x[0:]))
              
        if local == True:
            out1 = self.head(_out[:, 0])
            return out1, None, None
        recons_ = None
        if recons==True:
            recons_ = self.head_recons(_out[:, 1:])
        
        ftrs = self.head(_out)
        return ftrs[:, 0], ftrs[:, 1:], recons_
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASiT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ASiT(args)
