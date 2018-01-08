from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import os
import numpy as np

def plot(samples, name, output_path):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample, cmap='Greys_r')
    plt.savefig(os.path.join(output_path, '{0}.png'.format(name)), bbox_inches='tight')
    plt.close(fig)

def create_block_mask(shape):
    block_mask = np.ones(shape).astype('float32')
    inverse_block_mask = np.zeros(shape).astype('float32')
    #block area
    block_mask[:, 32:64, 32:64, :] = 0.0    
    inverse_block_mask[:, 32:64, 32:64, :] = 1.0
    
    return block_mask, inverse_block_mask

    
