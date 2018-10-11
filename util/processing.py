import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

def show_images(images, file_path='generated/save.png'):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
        plt.imsave(file_path, img.reshape([sqrtimg,sqrtimg]))


    return fig

def img_preprocessing(img):
    return img * 2 -1

def img_deprocessing(logits):
    return (logits + 1) * 0.5