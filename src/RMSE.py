import argparse
import os
import numpy as np

from utils import varia
from utils import SphericalHarmonics as sh
from mrtrix_io.io import load_mrtrix, save_mrtrix
from mrtrix_io.io.image import Image


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert dMRI data to a Rank-1 SH representation.')
    parser.add_argument('dwi1', help='Input dMRI data.')
    parser.add_argument('dwi2', help='Input dMRI data.')
    parser.add_argument('out', help='Output file.')
    parser.add_argument('--mask', help='Mask file.')

    args = parser.parse_args()

    data = load_mrtrix(args.dwi1)
    dwi1 = data.data
    dwi2 = load_mrtrix(args.dwi2).data
    mask = load_mrtrix(args.mask).data > .5
    
    rmse = varia.rmse(dwi1, dwi2, mask)
    save_mrtrix(args.out, Image(rmse, vox=data.vox))