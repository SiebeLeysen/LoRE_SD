# Provide functionality to scale the ODFs by the intra-axonal volume fractions
# or something similar.

# Add the source directory to the path
import sys, subprocess
sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD/src')
sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD')

import numpy as np
import argparse

from mrtrix_io.io import load_mrtrix, save_mrtrix
from mrtrix_io.io.image import Image
from scipy.ndimage import binary_erosion

def mask2contour(mask, output, vox):
    """
    Create a contour of the mask.

    Parameters:
    - mask (numpy.ndarray): The mask to create a contour of.
    - output (str): The output file to save the contour to.
    """
    contour = mask & ~binary_erosion(mask, structure=np.ones((5,5,5)))
    save_mrtrix(output, Image(contour.astype(np.float32), vox=vox))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mask', help='Mask file')
    parser.add_argument('output', help='Output file')

    args = parser.parse_args()

    # sample lesion on fine grid
    cmd = f'mrgrid {args.mask} regrid {args.output} -voxel .2 -interp nearest -force'
    subprocess.run(cmd, shell=True)

    # Read the image to find the mask boundaries
    regrid = load_mrtrix(args.output)
    x_min, x_max = np.where(regrid.data > 0)[0].min(), np.where(regrid.data > 0)[0].max()
    y_min, y_max = np.where(regrid.data > 0)[1].min(), np.where(regrid.data > 0)[1].max()
    z_min, z_max = np.where(regrid.data > 0)[2].min(), np.where(regrid.data > 0)[2].max()
    assert x_min < x_max and y_min < y_max and z_min < z_max
    cmd_crop = f'mrconvert {args.output} {args.output} -coord 0 {x_min}:1:{x_max} -coord 1 {y_min}:1:{y_max} -coord 2 {z_min}:1:{z_max} -force'
    subprocess.run(cmd_crop, shell=True)

    regrid = load_mrtrix(args.output)

    mask2contour(regrid.data > .5, args.output, regrid.vox)

    # Apply smoothing to the edge
    cmd = f'mrfilter {args.output} smooth {args.output} -force -extent 3'
    subprocess.run(cmd, shell=True)
