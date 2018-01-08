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
    
def creat_random_mask(shape):
    batch_size = shape[0]
    block_mask = np.ones(shape).astype('float32')
    inverse_block_mask = np.zeros(shape).astype('float32')
    x = np.random.randint(low=24, high=71, size=[batch_size])
    y = np.random.randint(low=24, high=71, size=[batch_size])
    w = np.random.randint(low=12, high=24, size=[batch_size])
    h = np.random.randint(low=12, high=24, size=[batch_size])
    idx = 0
    for x_idx, y_idx, w_idx, h_idx in zip(x, y, w, h):
        block_mask[idx, y_idx-h_idx:y_idx+h_idx, x_idx-w_idx:x_idx+w_idx, :] = 0.0
        inverse_block_mask[idx, y_idx-h_idx:y_idx+h_idx, x_idx-w_idx:x_idx+w_idx, :] = 1.0
        idx += 1
    
    return block_mask, inverse_block_mask

    
