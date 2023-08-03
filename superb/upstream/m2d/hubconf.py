# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/m2d/hubconf.py ]
#   Synopsis     [ the M2D torch hubconf ]
"""*********************************************************************************************"""

import os

from .expert import UpstreamExpert as _UpstreamExpert


def m2d_local(ckpt, model_config=None, *args, **kwargs):
    assert os.path.isfile(ckpt)
    if model_config is not None:
        assert os.path.isfile(model_config)
    if 'feature_d' not in kwargs:
        kwargs["feature_d"] = None
    return _UpstreamExpert(ckpt, model_config, *args, **kwargs)


def m2d_calcnorm(refresh=False, *args, **kwargs):
    """Upstream model entry for calculating normalization statistics for M2D on Superb.
    """

    if kwargs['ckpt'] is None:
        print('Set -i your-checkpoint. Exit now.')
        exit(-1)

    kwargs['ckpt'] = kwargs['ckpt'].split(',')[0]
    return m2d_local(*args, **kwargs)


def m2d(refresh=False, *args, **kwargs):
    """Upstream model entry for running M2D on Superb.
    Note:
        kwargs['ckpt']: "path-name-of-your-ckpt,dataset-mean,dataset-std".    
    """

    if kwargs['ckpt'] is None:
        print('Set "-k your-checkpoint.pth,dataset-mean,dataset-std". Exit now.')
        exit(-1)
    try:
        ckpt, norm_mean, norm_std = kwargs['ckpt'].split(',')
    except:
        print(f'Confirm your `ckpt`: {kwargs["ckpt"]}')
        exit(-1)

    kwargs['ckpt'] = ckpt
    norm_mean, norm_std = float(norm_mean), float(norm_std)
    print(' using checkpoint:', ckpt)
    print(' norm stats:', norm_mean, norm_std)
    return m2d_local(*args, norm_mean=norm_mean, norm_std=norm_std, **kwargs)
