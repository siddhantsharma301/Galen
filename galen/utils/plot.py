"""
Plotting Utilities for Galen
Author: Siddhant Sharma, 2019
"""
__author__ = "Siddhant Sharma"


import glob
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.utils import save_image


def plot_report(orig, recon, bbox, diff, cmap = mpl.cm.jet):
    """
    Plot report generated by Galen

    Args:
    * orig
        original image (`np.ndarray`)
    * recon
	reconstructed image (`np.ndarray`)
    * bbox 
	bounding box, if unhealthy (`np.ndarray`)
    * diff
	difference heatmap (`np.ndarray`)
    * cmap
        OPTIONAL
        Colormap for heatmap plot (`matplotlib.cm`)
        Default: `matplotlib.cm.jet`

    Returns:
    N/A, but plots and saves `matplotlib.pyplot.figure`
    """
    cmap = cmap
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    # Using dark mode is so extra 
    # plt.style.use('dark_background')

    fig = plt.figure(1)

    ax1 = fig.add_subplot(141)
    ax1.imshow(orig)
    ax1.set_title('Input Image')
    ax1.axis('off')

    ax2 = fig.add_subplot(142)
    ax2.imshow(recon)
    ax2.set_title('Reconstruction')
    ax2.axis('off')

    ax3 = fig.add_subplot(143)
    ax3.imshow(bbox)
    ax3.axis('off')
    ax3.set_title('BBox')

    ax4 = fig.add_subplot(144)
    im4 = ax4.imshow(diff, cmap = cmap, norm = norm)
    ax4.set_title('Heatmap')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size = '5%', pad=0.05)
    fig.colorbar(im4, cax = cax, orientation = 'vertical')
    ax4.axis('off')

    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    plt.savefig('galen/outputs/fig.png', bbox_inches = 'tight')


def reconstruction_grid(model, epoch, images, path='./training/gif/'):
    """
    Generate GIF images

    Args:
    * model
    	Galen model to use for reconstruction
    * epoch
        Number of epoch (for numbering) (int)
    * images
        Images to reconstruct and save (`torch.Tensor`)
    * path
    	OPTIONAL
    	path to store grid images (str)
    	Default: './training/gif/'

    Returns:
    N/A, but saves PNGs from reconstruction
    """
    with torch.no_grad():
        predictions, _, _ = model(images)
    predictions = predictions.cpu()
    save_image(predictions, path + 'epoch_{:04d}.png'.format(epoch), normalize=True, nrow=4)


def generate_gif(path='galen/training/gif/'):
    """
    Generate a GIF from training recostructions for 
    super cool visuals

    Args:
    * path
    	OPTIONAL
        path where grid images exist (str)
        Default: './training/gif/'

    Returns:
    N/A, but creates a .gif
    """

    anim_file = path + 'reconstructions.gif'
    
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(path + 'epoch_*.png')
        filenames = sorted(filenames)
        last = -1
        for i,filename in enumerate(filenames):
            frame = 2*(i**1)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
        
