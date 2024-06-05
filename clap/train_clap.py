"""M2D-CLAP Pre-training Script

M2D-CLAP: Masked Modeling Duo Meets CLAP for Learning General-purpose Audio-Language Representation
https://TBD
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import subprocess
import sys
import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import matplotlib.pyplot as plt

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from m2d import models_mae

from m2d.engine_pretrain_m2d import train_one_epoch_m2dx
from audio_dataset import SpectrogramDataset, get_files_no_sort, log_mixup_exp, pd
import common
from m2d.runtime_audio import RuntimeM2D


class CapEmbSpecDataset(torch.utils.data.Dataset):
    def __init__(self, base_folder, files_main, files_bg_noise, file_caption, crop_size, noise_ratio=0.0,
                 random_crop=True, n_norm_calc=10000) -> None:
        super().__init__()

        self.ds1 = SpectrogramDataset(folder=base_folder, files=files_main, crop_frames=crop_size[1],
                random_crop=random_crop, norm_stats=None, n_norm_calc=n_norm_calc//2)
        self.norm_stats = self.ds1.norm_stats  # for compatibility with SpectrogramDataset
        self.norm_std = self.ds1.norm_stats[1]
        self.ds1.norm_stats = (self.ds1.norm_stats[0], 1.0)

        if noise_ratio > 0.0:
            self.ds2 = SpectrogramDataset(folder=base_folder, files=files_bg_noise, crop_frames=crop_size[1],
                    random_crop=random_crop, norm_stats=None, n_norm_calc=n_norm_calc//2, repeat_short=True)
            self.ds2.norm_stats = (self.ds2.norm_stats[0], 1.0) # disable normalizion scaling in the ds2
        # for BG noise
        self.noise_ratio = noise_ratio
        self.bg_index = []
        # load captions
        self.capembs = []
        for capfile in file_caption.split(','):
            embs = np.load(capfile, allow_pickle=True).item()
            self.capembs.append(embs)
            print('Caption', Path(capfile).stem, '-> embeddings:', len(embs), ' expample keys:', list(embs.keys())[:5])
            assert list(self.capembs[0].values())[0].shape[-1] == list(self.capembs[-1].values())[0].shape[-1], 'All captions need to have the same number of embedding dimensions.'
        self.file_caption = file_caption
        self.caption_dim = list(self.capembs[0].values())[0].shape[-1]
        print('Using', len(self.capembs), 'caption definitions with feature dimension:', self.caption_dim)

    def __len__(self):
        return len(self.ds1)

    def get_random_caption(self, ytid):
        emb_list = []
        for capemb in self.capembs:
            if ytid in capemb:
                cur = capemb[ytid]
                cur = [cur] if len(cur.shape) == 1 else [cur[i] for i in range(cur.shape[0])]  # prepare list of captions
                emb_list.extend(cur)
        # for debug
        # if len(emb_list) > 1:
        #     print(ytid, 'has multiple captions:', len(emb_list))
        return random.choice(emb_list)

    def __getitem__(self, index, fixed_noise=False):
        # load index sample
        clean = self.ds1[index]
        if self.noise_ratio > 0.0:
            # load random noise sample ### , while making noise floor zero
            noise = self.ds2[index if fixed_noise else self.get_next_bgidx()]
            # mix
            mixed = log_mixup_exp(noise, clean, self.noise_ratio) if self.noise_ratio < 1.0 else noise
        else:
            mixed = clean
        # finalize normalization. clean and noise were averaged to zero. the following will scale to 1.0 using ds1 std.
        mixed = mixed / self.norm_std

        # load sample's caption
        ytid = self.ds1.df.file_name.values[index].split('/')[-1][:11]
        caption = self.get_random_caption(ytid)

        return mixed, caption

    def get_next_bgidx(self):
        if len(self.bg_index) == 0:
            self.bg_index = torch.randperm(len(self.ds2)).tolist()
            # print(f'Refreshed the bg index list with {len(self.bg_index)} items: {self.bg_index[:5]}...')
        return self.bg_index.pop(0)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(crop_frames={self.ds1.crop_frames}, '
        format_string += f'folder_main={self.ds1.df.file_name.values[0].split("/")[0]}, '
        if self.noise_ratio > 0.: format_string += f'folder_bg={self.ds2.df.file_name.values[0].split("/")[0]}, '
        format_string += f'caption={self.file_caption})'
        return format_string


def build_captioned_dataset(cfg):
    # get files and inflate the number of files (by repeating the list) if needed
    files_main = get_files_no_sort(cfg.csv_main)
    files_bg = get_files_no_sort(cfg.csv_bg_noise) if cfg.noise_ratio > 0. else []

    ds = CapEmbSpecDataset(
        base_folder=cfg.data_path, files_main=files_main,
        files_bg_noise=files_bg,
        file_caption=cfg.file_caption,
        crop_size=cfg.input_size,
        noise_ratio=cfg.noise_ratio,
        random_crop=True)

    val_ds = SpectrogramDataset(folder=cfg.data_path, files=get_files_no_sort(cfg.csv_val), crop_frames=cfg.input_size[1], random_crop=True) \
        if cfg.csv_val else None

    return ds, val_ds


def get_args_parser():
    parser = argparse.ArgumentParser('Masked Modeling Duo (M2D) + CLAP pre-training', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--eval_after', default=50, type=int)
    parser.add_argument('--save_freq', default=100, type=int)
    parser.add_argument('--feature_eval_freq', default=10, type=int, help='Feature label-free evaluation frequency.')
    parser.add_argument('--stop_at', default=-1, type=int)

    # Model parameters
    parser.add_argument('--model', default='m2d_clap_vit_base', type=str, metavar='MODEL', help='Model name.')
    parser.add_argument('--decoder_depth', type=int, default=8, metavar='DD', help='Model decoder depth.')

    parser.add_argument('--input_size', default='80x608', type=str, help='images input size')
    parser.add_argument('--patch_size', default='16x16', type=str, help='patch size')
    parser.add_argument('--sr', default='16k', type=str, metavar='SR', help='Sampling rate of the input audio.')

    parser.add_argument('--mask_ratio', default=0.7, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--ema_decay_init', default=0.99995, type=float,
                        help='Initial EMA decay parameter.')
    parser.add_argument('--ema_decay', default=0.99999, type=float,
                        help='EMA decay parameter.')
    parser.add_argument('--loss_fn', default='norm_mse', type=str,
                        help='loss function: mse or norm_mse.')
    parser.add_argument('--loss_m2d', default=1., type=float, help='Loss of M2D masked prediction')
    parser.add_argument('--loss_off', default=.01, type=float, help='Loss of offline target')
    parser.add_argument('--target_layers', default='', type=str,
                        help='Experimental: layers to calculate target representations.')
    parser.add_argument('--bf16', action='store_true')

    parser.add_argument('--no_norm_pix_loss', action='store_false', dest='norm_pix_loss',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)

    parser.add_argument('--cont_mask', default=0, type=int,
                        help='Use random 1-d (continuous) masking scheme. 0:off, 0<:# of frames to make them continuous.')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--clip_grad', type=float, default=3.0, metavar="NORM",
                        help="Clip gradient norm (default: None, no clipping)")
    parser.add_argument('--optim', default='adamw', type=str, help='Optimizer adam or sdg')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=3e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='data', type=str,
                        help='dataset path')
    parser.add_argument('--csv_main', default='data/files_audioset.csv', type=str,
                        help='A CSV file to list sample files in the main dataset')
    parser.add_argument('--csv_bg_noise', default='', type=str,
                        help='A CSV file to list sample files in the BG noise dataset')
    parser.add_argument('--file_caption', default='data/capemb_GTEbase_Audo_A_C_D.npy', type=str,
                        help='A npy file for the caption')
    parser.add_argument('--csv_val', default='', type=str,
                        help='A CSV file to list validation sample files')
    parser.add_argument('--min_ds_size', default=10000, type=int,
                        help='Inflate the size of the smaller dataset to the desired size')
    parser.add_argument('--norm_stats', default='None', type=str,  # Will be computed runtime.
                        help='dataset normalization stats')
    parser.add_argument('--noise_ratio', default=0., type=float,
                        help='Noise mixing ratio')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--force_start_epoch', default=0, type=int, metavar='N',  # 0=always reset start epoch to 0 even if using --resume
                        help='start epoch for resuming')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank' if torch.__version__ >= "2.0.0" else '--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def ema_decay_sched(step, total_steps, ema_decay_init, ema_decay):
    interp = step / (total_steps - 1)
    tau = ema_decay_init + (ema_decay - ema_decay_init) * interp
    return tau


def get_optim(args, param_groups):
    if args.optim == 'adamw':
        return torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    elif args.optim == 'sgd':
        return torch.optim.SGD(param_groups, args.lr, momentum=0.9, weight_decay=0)
    assert False, f'Unsupported optimizer {args.optim}'


def load_model(args, model_without_ddp, optimizer, loss_scaler, delta_epoch=1, strict=True):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=strict)
        print("Resume checkpoint %s" % args.resume)
        if strict == True and 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + delta_epoch
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, dataset_val = build_captioned_dataset(args)
    print(dataset_train, dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    common.PrintLogger(f'{args.log_dir}/console.txt')
    print(args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # for validation loss
    if args.csv_val != '':
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        print("Sampler_val = %s" % str(sampler_val))
        data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    else:
        data_loader_val = None
    
    # define the model
    model = models_mae.__dict__[args.model](img_size=args.input_size, patch_size=args.patch_size, decoder_depth=args.decoder_depth,
        norm_pix_loss=args.norm_pix_loss, loss_type=args.loss_fn, target_layers=args.target_layers, loss_m2d=args.loss_m2d, loss_off=args.loss_off, norm_stats=dataset_train.norm_stats,
        off_emb_dim=dataset_train.caption_dim, clip_args=[global_rank, num_tasks])

    if args.cont_mask > 0:
        model.set_random_1d_mask(args.cont_mask)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    org_args_lr = args.lr
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / args.eff_batch_size) if org_args_lr is None else 'base lr: not effective')
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % args.eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    try:
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    except:
        print(' (for compatibility with timm) Switched add_weight_decay() to param_groups_weight_decay()')
        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = get_optim(args, param_groups)
    print(optimizer)
    loss_scaler = NativeScaler()

    load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, delta_epoch=0, strict=False)

    if args.force_start_epoch >= 0:
        args.start_epoch = args.force_start_epoch

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    last_subprocess = None
    for epoch in range(args.start_epoch, args.epochs):
        epoch1 = epoch + 1
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch_m2dx(
            model, None, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            partial(ema_decay_sched, total_steps=len(data_loader_train) * args.epochs,
                ema_decay_init=args.ema_decay_init, ema_decay=args.ema_decay),
            val_loader=data_loader_val,
            log_writer=log_writer,
            do_analysis=(epoch1 % args.feature_eval_freq == 0),
            autocast_args=dict(dtype=torch.bfloat16) if args.bf16 else {},
            args=args
        )

        if args.output_dir and (epoch1 % args.save_freq == 0 or epoch1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch1)
            # run the external evaluator
            if args.eval_after <= epoch1 and epoch1 < args.epochs and misc.is_main_process():
                abspath = Path(f'{args.output_dir}/checkpoint-{epoch1}.pth').absolute()
                print('quick_eval', abspath)
                last_subprocess = subprocess.Popen(['/bin/bash', './quick_eval.sh', abspath])

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if args.stop_at > 0 and epoch1 >= args.stop_at:
            if last_subprocess is not None:
                last_subprocess.wait()
            print(f'Stop training by reaching args.stop_at epoch: {args.stop_at}')
            exit(0)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    del model_without_ddp, model, data_loader_train, optimizer, loss_scaler
    if misc.is_main_process():
        abspath = Path(f'{args.output_dir}/checkpoint-{epoch1}.pth').absolute()
        subprocess.call(['/bin/bash', './all_eval.sh', abspath])
    return epoch1


arg_conf_defaults = {
    'csv_main': ('data/files_audioset.csv', 'M', 'path'),
    'csv_bg_noise': ('', 'D', 'path'),
    'file_caption': ('', 'C', 'path'),
    'ema_decay_init': (0.99995, 'ema', 'z'),
    'ema_decay': (0.99999, 'ed', 'z'),
    'decoder_depth': (8, 'dd', 'asis'),
    'mask_ratio': (0.7, 'mr', 'z'),
    'seed': (0, 's', 'asis'),
    'norm_pix_loss':  (True, '~N', 'b'),
    'loss_fn': ('norm_mse', 'L', 'head'),
    'optim': ('adamw', 'O', 'asis'),
    'warmup_epochs': (20, 'wu', 'asis'),
    'blr': (3e-4, 'blr', 'z'),
    'lr': (None, 'lr', 'z'),
    'eff_batch_size': (2048, 'bs', 'asis'),
    'accum_iter': (1, 'a', 'asis'),
    'loss_m2d': (1.0, 'lm', 'z'),
    'loss_off': (0.01, 'lo', 'z'),
    'noise_ratio': (0.0, 'nr', 'z'),
    'min_ds_size': (10000, 'dn', 'asis'),
    'cont_mask': (0, 'C', 'asis'),
    'epochs': (0, '-e', 'asis'),
}


def complete_args():
    args = get_args_parser()
    args = args.parse_args()
    _input_size, _patch_size = args.input_size, args.patch_size
    args.input_size = [int(x) for x in args.input_size.split('x')]
    args.patch_size = [int(x) for x in args.patch_size.split('x')]
    args.norm_stats = eval(args.norm_stats) if args.norm_stats else None

    misc.init_distributed_mode(args)

    args.eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if not args.output_dir:
        args.output_dir = f'{args.model}-{_input_size}p{_patch_size}p{args.sr}'
        args.output_dir += f'-{common.get_timestamp()[:6]}-{common.arg_conf_str(args, defaults=arg_conf_defaults)}'

    if not args.log_dir:
        args.log_dir = args.output_dir
    args.target_layers = None if args.target_layers == '' else eval(args.target_layers)
    return args


if __name__ == '__main__':
    args = complete_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
