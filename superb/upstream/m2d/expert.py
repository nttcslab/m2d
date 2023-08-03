# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/m2d/expert.py ]
#   Synopsis     [ the Masked Modeling Duo (M2D) wrapper ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import math
#-------------#
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#-------------#
from .m2d.m2d.runtime_audio import RuntimeM2D


class RunningMean:
    """Running mean calculator for arbitrary axis configuration.
    Borrowed from https://github.com/nttcslab/byol-a/blob/master/v2/byol_a2/augmentations.py#L147
    """

    def __init__(self, axis):
        self.n = 0
        self.axis = axis

    def put(self, x):
        # https://math.stackexchange.com/questions/106700/incremental-averageing
        self.n += 1
        if self.n == 1:
            self.mu = x.mean(self.axis, keepdims=True)
        else:
            self.mu += (x.mean(self.axis, keepdims=True) - self.mu) / self.n

    def __call__(self):
        return self.mu

    def __len__(self):
        return self.n


class RunningVariance:
    """Calculate mean/variance of tensors online.
    Thanks to https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    Borrowed from https://github.com/nttcslab/byol-a/blob/master/v2/byol_a2/augmentations.py#L147
    """

    def __init__(self, axis, mean):
        self.update_mean(mean)
        self.s2 = RunningMean(axis)

    def update_mean(self, mean):
        self.mean = mean

    def put(self, x):
        self.s2.put((x - self.mean) **2)

    def __call__(self):
        return self.s2()

    def std(self):
        return self().sqrt()


class RunningNorm(nn.Module):
    """Online Normalization using Running Mean/Std.
    Borrowed from https://github.com/nttcslab/byol-a/blob/master/v2/byol_a2/augmentations.py#L147
    This module will only update the statistics up to the specified number of epochs.
    After the `max_update_epochs`, this will normalize with the last updated statistics.
    Args:
        epoch_samples: Number of samples in one epoch
        max_update_epochs: Number of epochs to allow update of running mean/variance.
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, epoch_samples, max_update_epochs=10, axis=[1, 2]):
        super().__init__()
        self.max_update = epoch_samples * max_update_epochs
        self.ema_mean = RunningMean(axis)
        self.ema_var = RunningVariance(axis, 0)
        self.reported = False

    def forward(self, image):
        if len(self.ema_mean) < self.max_update:
            self.ema_mean.put(image)
            self.ema_var.update_mean(self.ema_mean())
            self.ema_var.put(image)
            self.mean = self.ema_mean()
            self.std = torch.clamp(self.ema_var.std(), torch.finfo().eps, torch.finfo().max)
        elif not self.reported:
            self.reported = True
            logger.info(f'\n*** Running Norm has finished updates over {self.max_update} times, using the following stats from now on. ***\n  mean={float(self.mean.view(-1))}, std={float(self.std.view(-1))}')
            logger.info(f'*** Please use these statistics in your model. EXIT... ***\n')
            exit(-1)
        return ((image - self.mean) / self.std)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(max_update={self.max_update},axis={self.ema_mean.axis})'
        return format_string


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(nn.Module):
    """
    The M2D wrapper
    """

    def __init__(
        self,
        ckpt: str,
        model_config: str,
        feature_d: int,
        window_secs: float = (160 * 16) / 16000,
        stride_secs: float = (160 * 16) / 16000,
        norm_mean: float = None,  # Has to be a float value to continue training.
        norm_std: float = None,  # The same as above.
        **kwargs,
    ):
        super(UpstreamExpert, self).__init__()

        # Normalizer
        if norm_mean is None or norm_std is None:
            # ** CAUTION **
            # ** Please note that here we calculate statistics using RunningNorm and will exit early in the training. **
            # ** CAUTION **
            self.norm = RunningNorm(epoch_samples=10_000, max_update_epochs=1, axis=[0, 1, 2, 3])  # Use single scalar mean/std values.
        else:
            print(f'*** Using normalization statistics: mean={norm_mean}, std={norm_std} ***')
            self.norm = lambda x: (x - norm_mean) / norm_std
        

        # Load pretrained weights.
        self.model = RuntimeM2D(weight_file=ckpt)

        # attributes
        self.output_dim = self.model.cfg.feature_d
        self.max_input_length = 1024  # self.model.cfg.input_size[1]

    # Interface
    def get_output_dim(self):
        return self.output_dim

    # Interface
    def get_downsample_rates(self, key: str) -> int:
        return 160 * self.model.cfg.patch_size[1]  # hop_size x time frames

    def to_feature(self, batch_audio):
        x = self.model.to_spec(batch_audio)
        x = (x + torch.finfo().eps).log()
        return x.unsqueeze(1) #.to(device)

    # Interface
    def forward(self, wavs):
        """
        Args:
            wavs:
                list of unpadded wavs [wav1, wav2, ...]
                each wav is in torch.FloatTensor with sample rate 16000
                and already put in the device assigned by command-line args

        Return:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args
        """
        wavs = pad_sequence(wavs, batch_first=True)
        features = self.to_feature(wavs)
        # normalize
        features = self.norm(features)
        # encode
        layered_features = self.model.encode_lms(features, return_layers=True)
        return {
            "last_hidden_state": layered_features[-1],
            "hidden_states": layered_features,
        }

