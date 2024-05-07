"""A utility for M2D pre-trained weight files.
This script converts an M2D weight to an encoder-only weight, resulting in a much smaller weight (1.6G to 326M).

Usage: python [this script] [source checkpoint file] [output checkpoint file]
"""

import torch
from pathlib import Path
import sys
sys.path.append('examples')
from portable_m2d import PortableM2D

src_file = sys.argv[1]
dest_file = sys.argv[2]

if not Path(src_file).stem.startswith('checkpoint'):
    print(f' **WARNING** Do not use this converter for the fine-tuned weights. HEAD WEIGHTS WILL BE LOST.')

# Load the weight. All the parameters not used in the encoder-only model will be deleted.
# The parameter `norm_stats` will be added if the weight does not have it. i.e., Old weights.
model = PortableM2D(src_file)

# Save the weights.
Path(dest_file).parent.mkdir(exist_ok=True, parents=True)
torch.save(model.backbone.state_dict(), dest_file)
print(f'Saved {dest_file}.')
