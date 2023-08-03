# Masked Modeling Duo (M2D)

import datetime
import hashlib
import sys
import re


class PrintLogger(object):
    def __init__(self, logfile):
        self.stdout = sys.stdout
        self.log = open(logfile, 'a')
        sys.stdout = self

    def write(self, message):
        self.stdout.write(message)
        self.log.write(message)  

    def flush(self):
        self.stdout.flush()


def get_timestamp():
    """ex) Outputs 202104220830"""
    return datetime.datetime.now().strftime('%y%m%d%H%M')


def hash_text(text, L=128):
    hashed = hashlib.shake_128(text.encode()).hexdigest(L//2 + 1)
    return hashed[:L]


def short_model_desc(model, head_len=5, tail_len=1):
    text = repr(model).split('\n')
    text = text[:head_len] + ['  :'] + (text[-tail_len:] if tail_len > 0 else [''])
    return '\n'.join(text)


def prmstr_z(p):
    return str(p).replace('.0', '').replace('0.', '.')

def prmstr_zz(prm):
    ps = [prmstr_z(p) for p in prm]
    return '-'.join(ps)


conf_defaults={
    'dataset': ('data/files_audioset.csv', 'D', 'path'),
    'ema_decay_init': (0.99995, 'ema', 'z'),
    'ema_decay': (0.99999, 'ed', 'z'),
    'decoder_depth': (8, 'dd', 'asis'),
    'mask_ratio': (0.7, 'mr', 'z'),
    'seed': (0, 's', 'asis'),
    'norm_pix_loss':  (True, '~N', 'b'),
    'loss_fn': ('norm_mse', 'L', 'head'),
    'optim': ('adamw', 'O', 'asis'),
    'blr': (3e-4, 'blr', 'z'),
    'lr': (None, 'lr', 'z'),
    'eff_batch_size': (2048, 'bs', 'asis'),
    'accum_iter': (1, 'a', 'asis'),
}


def arg_conf_str(args, defaults=conf_defaults):
    confstr = ''
    for k in defaults:
        try:
            arg_value = eval('args.' + k)
        except:
            continue # no parameter k for the run.
        if arg_value == defaults[k][0]:
            continue
        arg_key, value_format = defaults[k][1:]
        value = str(arg_value)
        if value_format == 'z':
            value = prmstr_z(arg_value)
        elif value_format == 'zz':
            value = prmstr_zz(arg_value)
        elif value_format == 'b':
            value = '' # nothing to add
        elif value_format == 'head':
            value = value[:1]
        elif value_format == 'head_':
            value = ''.join([v[:1] for v in value.split('_')])
        elif value_format == 'path':
            value = ''.join([v[:1] for v in re.split(r'_|/', value)])
        confstr += arg_key + value
    return confstr
