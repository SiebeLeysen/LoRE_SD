import argparse

import numpy as np
from scipy.optimize import lsq_linear

from lore_sd.mrtrix_io.io import load_mrtrix, save_mrtrix
from lore_sd.mrtrix_io.io.image import Image

def main(input, weights, mask=None):
    ad, rd = np.linspace(0, 4e-3, weights.shape[-2]), np.linspace(0, 4e-3, weights.shape[-1])
    ad_gte_rd = ad[:, None] >= rd[None, :]

    if mask is not None:
        input_vector = input[mask]
        input_weights = weights[mask][..., ad_gte_rd]

    else:
        input_vector = input.flatten()
        input_weights = weights[..., ad_gte_rd]

    result = lsq_linear(input_weights, input_vector, bounds=(0, 1))
    print(f'Optimization finished...')
    estimated_weights = np.zeros((len(ad), len(rd)))
    estimated_weights[ad_gte_rd] = result.x

    estimated_contrast = np.einsum('ab, ...ab -> ...', estimated_weights, weights)

    return estimated_weights, estimated_contrast




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate the contrast matrix that best reconstructs the input contrast.')
    parser.add_argument('input_contrast', help='Input contrast')
    parser.add_argument('input_weights', help='Input weights of the response function representation.')
    parser.add_argument('output_matrix', help='Output matrix. Should be a .txt file.')
    parser.add_argument('output_contrast', help='Output contrast. Should be a .mif file.')
    parser.add_argument('--mask', help='Mask to be used in the estimation.')

    args = parser.parse_args()
    
    print(f'Loading data...')
    input_contrast = load_mrtrix(args.input_contrast).data
    input_weights = load_mrtrix(args.input_weights).data
    mask = load_mrtrix(args.mask).data > .5 if args.mask is not None else None

    input_vox = load_mrtrix(args.input_contrast).vox

    print('Data loaded. Starting optimization...')

    estimated_weights, estimated_contrast = main(input_contrast, input_weights, mask)

    print('Optimization finished. Saving results...')

    np.savetxt(args.output_matrix, estimated_weights)
    save_mrtrix(args.output_contrast, Image(estimated_contrast, vox=input_vox, comments=f'Estimated contrast matrix from {args.input_contrast}'))

    print('Results saved.')
