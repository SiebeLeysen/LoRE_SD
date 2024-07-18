import argparse
import numpy as np
import os

from optimisation import contrasts
from mrtrix_io.io import load_mrtrix, save_mrtrix
from mrtrix_io.io.image import Image

def main():
    parser = argparse.ArgumentParser(description='Derive the intra-axonal, extra-axonal and free water contrast '
                                                 'given an image of gaussian fractions')
    parser.add_argument('input_fractions', type=str, help='The gaussian fractions')
    parser.add_argument('output_dir', type=str, help='The output directory')
    args = parser.parse_args()

    # Your code here
    data = load_mrtrix(args.input_fractions)
    fractions = data.data
    vox = data.vox

    # img_shape = fractions.shape[:-2]
    num_ad, num_rd = fractions.shape[-2:]

    ad_range = np.linspace(0, 4e-3, num_ad)
    rd_range = np.linspace(0, 4e-3, num_rd)

    intra_ax_contrast = contrasts.get_contrast(fractions, ad_range, rd_range, contrasts.intra_axonal_contrast)
    extra_ax_contrast = contrasts.get_contrast(fractions, ad_range, rd_range, contrasts.extra_axonal_contrast)
    free_water_contrast = contrasts.get_contrast(fractions, ad_range, rd_range, contrasts.free_water_contrast)

    save_mrtrix(os.path.join(args.output_dir, 'intra_axonal_contrast.mif'), Image(intra_ax_contrast, vox=vox, comments=[]))
    save_mrtrix(os.path.join(args.output_dir, 'extra_axonal_contrast.mif'), Image(extra_ax_contrast, vox=vox, comments=[]))
    save_mrtrix(os.path.join(args.output_dir, 'free_water_contrast.mif'), Image(free_water_contrast, vox=vox, comments=[]))

if __name__ == '__main__':
    main()