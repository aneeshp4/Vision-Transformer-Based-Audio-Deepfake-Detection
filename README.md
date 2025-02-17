# ASiT: Audio Spectrogram vIsion Transformer for General Audio Representation

![](ASiT.png)

This repository contains the official PyTorch self-supervised pretraining, finetuning, and evaluation codes for 
[ASiT](https://arxiv.org/abs/2211.13189): Audio Spectrogram vIsion Transformer for General Audio Representation.

The finetuning strategy is adopted from [AST](https://github.com/YuanGongND/ast) 

# Self-supervised pre-training
> python -m torch.distributed.launch --nproc_per_node=4 --use_env main_ASiT.py --batch_size 20 --epochs 100 --data_path 'path/to/audio/files' --data-train 'path/to/json/file'

Self-supervised pre-trained models using ASiT can be downloaded from [here](https://drive.google.com/file/d/11eaOU40jonpYZ3u_XI-XUSSWclv8qeR7/view?usp=drive_link)

# Data Preparation
We mainly employed AudioSet for ASiT pre-training which contains YouTube videos. Please follow [link](https://research.google.com/audioset/download.html) to download and process AudioSet data.

If you use this code for a paper, please cite:

```
@article{atito2022asit,

  title={ASiT: Audio Spectrogram vIsion Transformer for General Audio Representation},
  
  author={Atito, Sara and Awais, Muhammad and Wang, Wenwu and Plumbley, Mark D and Kittler, Josef},
  
  journal={arXiv preprint arXiv:2211.13189},
  
  year={2022}
  
}
```
