# This file will take two files with ODFs as input and write the angular correlation coefficient (Anderson, 2005) to a file.

import argparse

import numpy as np

from mrtrix_io.io import load_mrtrix, save_mrtrix
from mrtrix_io.io.image import Image
from utils import SphericalHarmonics as sh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the angular correlation coefficient between two ODFs.')
    parser.add_argument('odf1', type=str, help='The first ODF file.')
    parser.add_argument('odf2', type=str, help='The second ODF file.')
    parser.add_argument('mask', type=str, help='The mask file.')
    parser.add_argument('output', type=str, help='The output file.')

    args = parser.parse_args()

    data1 = load_mrtrix(args.odf1)
    data2 = load_mrtrix(args.odf2)
    mask = load_mrtrix(args.mask).data > .5

    # Check if the last dimension is the same
    if data1.shape[-1] != data2.shape[-1]:
        raise ValueError('The last dimension of the two input files must be the same. Got {} and {}.'.format(data1.shape[-1], data2.shape[-1]))

    # Calculate the angular correlation coefficient
    angular_correlation = sh.angularCorrelation(data1.data[mask], data2.data[mask])
    out = np.zeros(mask.shape)
    out[mask] = angular_correlation

    # Save the result
    save_mrtrix(args.output, Image(out, vox=data1.vox,
                                   comments=[f'Angular correlation coefficient between '
                                             f'{args.odf1.split("/")[-1]} and {args.odf2.split("/")[-1]}']))