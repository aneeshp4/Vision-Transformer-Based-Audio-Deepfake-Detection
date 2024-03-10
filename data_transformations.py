import torchaudio
import numpy as np
import torch
import torch.nn.functional
import random
from numpy.random import randint

import torchaudio_augmentations as ADA

 
def GMML_replace_list(samples, corrup_prev, masks_prev, drop_type='noise', max_replace=0.35, align=16):

    rep_drop = 1 if drop_type == '' else ( 1 / ( len(drop_type.split('-')) + 1 ) )

    n_imgs = samples.size()[0] #this is batch size, but in case bad inistance happened while loading
    samples_aug = samples.detach().clone()
    masks = torch.zeros_like(samples_aug)
    for i in range(n_imgs):
        idx_rnd = randint(0, n_imgs)
        if random.random() < rep_drop:
            samples_aug[i], masks[i] = GMML_replace_patches(samples_aug[i], samples[idx_rnd], max_replace=max_replace, align=align)
        else:
            samples_aug[i], masks[i] = corrup_prev[i], masks_prev[i]

    return samples_aug, masks


def GMML_replace_patches(X, X_rep=None, drop_type='noise', max_replace=0.7, align=16, max_block_sz=0.3):
    #######################
    # max_replace: percentage of image to be replaced
    # align: align corruption with the patch sizes
    # max_block_sz: percentage of the maximum block to be dropped
    #######################

    np.random.seed()
    C, H, W = X.size()
    n_drop_pix = np.random.uniform(min(0.5, max_replace), max_replace)*H*W
    mx_blk_height = int(H*max_block_sz)
    mx_blk_width = int(W*max_block_sz)

    align = max(1, align)

    mask = torch.zeros_like(X)
    drop_t = np.random.choice(drop_type.split('-'))

    while mask[0].sum() < n_drop_pix:

        ####### get a random block to replace 
        rnd_r = ( randint(0, H-align) // align ) * align
        rnd_c = ( randint(0, W-align) // align ) * align

        rnd_h = min(randint(align, mx_blk_height), H-rnd_r)
        rnd_h = round( rnd_h / align ) * align
        rnd_w = min(randint(align, mx_blk_width), W-rnd_c)
        rnd_w = round( rnd_w / align ) * align

        if X_rep is not None:
            X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = X_rep[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w].detach().clone()
        else:
            if drop_t == 'noise':
                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = torch.empty((C, rnd_h, rnd_w), dtype=X.dtype, device=X.device).normal_()
            elif drop_t == 'zeros':
                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = torch.zeros((C, rnd_h, rnd_w), dtype=X.dtype, device=X.device)
            else:
                ####### get a random block to replace from
                rnd_r2 = (randint(0, H-rnd_h) // align ) * align
                rnd_c2 = (randint(0, W-rnd_w) // align ) * align

                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = X[:, rnd_r2:rnd_r2+rnd_h, rnd_c2:rnd_c2+rnd_w].detach().clone()

        mask[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = 1

    return X, mask


class DataAugmentation(object):
    def __init__(self, args):
        
        # for corruption
        self.drop_perc = args.drop_perc
        self.drop_type = args.drop_type
        self.drop_align = args.drop_align
        
        self.data_mean = args.data_mean
        self.data_std = args.data_std

        self.num_crops = args.num_crops
        self.num_crops_local = args.num_crops_local
        
        self.num_frames = args.num_frames
        self.num_frames_local = args.num_frames_local
        self.num_mel_bins = args.num_mel_bins
        
        # number of seconds to be used
        self.sample_rate = args.sample_rate
        self.num_samples = self.sample_rate * args.secs_per_crop
        self.num_samples_local = self.sample_rate * args.secs_per_crop_local
        
        self.transform = ADA.ComposeMany([ADA.RandomResizedCrop(n_samples=self.num_samples)], num_augmented_samples=self.num_crops)
        self.transform_local = ADA.ComposeMany([ADA.RandomResizedCrop(n_samples=self.num_samples_local)], num_augmented_samples=self.num_crops_local)
        

    def GMML_drop_rand_patches(self, X, max_block_sz=0.3):
        #######################
        # max_replace: percentage of image to be replaced
        # align: align corruption with the patch sizes
        # max_block_sz: percentage of the maximum block to be dropped
        #######################
       
        np.random.seed()    
        C, H, W = X.size()
        n_drop_pix = np.random.uniform(min(0.5, self.drop_perc), self.drop_perc)*H*W
        mx_blk_height = int(H*max_block_sz)
        mx_blk_width = int(W*max_block_sz)
        
        align = max(1, self.drop_align)
        
        mask = torch.zeros_like(X)
        
        while mask.sum() < n_drop_pix:
            
            ####### get a random block to replace 
            rnd_r = ( randint(0, H-align) // align ) * align
            rnd_c = ( randint(0, W-align) // align ) * align

            rnd_h = min(randint(align, mx_blk_height), H-rnd_r)
            rnd_h = round(rnd_h / align) * align
            rnd_w = min(randint(align, mx_blk_width), W-rnd_c)
            rnd_w = round(rnd_w / align) * align

            if self.drop_type == 'noise':
                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = torch.empty((C, rnd_h, rnd_w), 
                                                                    dtype=X.dtype, device=X.device).normal_(mean=self.data_mean, std=self.data_std)
                mask[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = 1 
            elif self.drop_type == 'zeros':
                X[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = torch.zeros((C, rnd_h, rnd_w), dtype=X.dtype, device=X.device)
                mask[:, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = 1 
            elif self.drop_type == 'time':
                X[:, rnd_r:rnd_r+rnd_h, :] = torch.zeros((C, rnd_h, W), dtype=X.dtype, device=X.device)
                mask[:, rnd_r:rnd_r+rnd_h, :] = 1 
            elif self.drop_type == 'freq':
                X[:, :, rnd_c:rnd_c+rnd_w] = torch.zeros((C, H, rnd_w), dtype=X.dtype, device=X.device)
                mask[:, :, rnd_c:rnd_c+rnd_w] = 1 
            else:
                print('Not Implemented!!')

        return X, mask
    
    def _wav2fbank(self, waveform, n_frames_):
        

        waveform = (waveform - waveform.mean())

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=self.sample_rate, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.num_mel_bins, dither=0.0, frame_shift=10)

        n_frames = fbank.shape[0]
        p = n_frames_ - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:n_frames_, :]

        return fbank
        
    def __call__(self, waveform):
        
        #pad waveform if less than num_samples
        if waveform.size()[1] < self.num_samples:
            waveform = torch.cat( (waveform, torch.zeros(1, self.num_samples-waveform.size()[1]) ), dim=-1)
            
        waveform_global = self.transform(waveform)
        

        # loop over crops
        fbanks, corr, masks = [], [], []
        for wvfrm in waveform_global:
            fbank = self._wav2fbank(wvfrm, self.num_frames)
    
            fbank = (fbank - self.data_mean) / (self.data_std*2)
            fbank = fbank.unsqueeze(0)
            
            # clean crop 
            fbanks.append(fbank)
            
            # corrupted and masked
            audio_corr = fbank.detach().clone()
            audio_mask = torch.zeros_like(audio_corr)
            if self.drop_perc > 0:
                audio_corr, audio_mask = self.GMML_drop_rand_patches(audio_corr)
                
            corr.append(audio_corr)
            masks.append(audio_mask)
            
        if self.num_crops_local > 0:
            waveform_local = self.transform_local(waveform)
            for wvfrm in waveform_local:
                fbank = self._wav2fbank(wvfrm, self.num_frames_local)
        
                fbank = (fbank - self.data_mean) / (self.data_std*2)
                fbank = fbank.unsqueeze(0)
                
                # clean crop 
                fbanks.append(fbank)
        
            
        return fbanks, corr, masks
