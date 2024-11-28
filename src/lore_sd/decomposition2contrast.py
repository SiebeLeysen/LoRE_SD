import argparse
import numpy as np
import os
import sys

# sys.path.append('/DIRECTORY_TO_REPO/LoRE_SD/LoRE-SD/src')
# sys.path.append('/DIRECTORY_TO_REPO/LoRE_SD/LoRE-SD')

from lore_sd.optimisation import contrasts
from lore_sd.mrtrix_io.io import load_mrtrix, save_mrtrix
from lore_sd.mrtrix_io.io.image import Image

def main():
    parser = argparse.ArgumentParser(description='Derive the intra-axonal, extra-axonal and free water contrast '
                                                 'given an image of gaussian fractions')
    parser.add_argument('input_fractions', type=str, help='The gaussian fractions')
    parser.add_argument('input_response', type=str, help='The response function')
    parser.add_argument('output_dir', type=str, help='The output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Your code here
    data = load_mrtrix(args.input_fractions)
    rf = load_mrtrix(args.input_response).data
    fractions = data.data
    vox = data.vox

    # img_shape = fractions.shape[:-2]
    num_ad, num_rd = fractions.shape[-2:]

    ad_range = np.linspace(0, 4e-3, num_ad)
    rd_range = np.linspace(0, 4e-3, num_rd)

    intra_ax_contrast = contrasts.get_contrast(fractions, ad_range, rd_range, contrasts.intra_axonal_contrast, True, 10)
    extra_ax_contrast = contrasts.get_contrast(fractions, ad_range, rd_range, contrasts.extra_axonal_contrast, True, 10)
    free_water_contrast = contrasts.get_contrast(fractions, ad_range, rd_range, contrasts.free_water_contrast)

    rf_anisotropy = np.nan_to_num(np.linalg.norm(rf[..., 1:,1:], axis=(-1, -2)), nan=0, posinf=0, neginf=0)
    rf_anisotropy /= np.max(rf_anisotropy)
    normed_anisotropy = np.nan_to_num(np.linalg.norm(rf[..., -1:,1:]/rf[...,-1:,:1], axis=(-1, -2)), nan=0, posinf=0, neginf=0)
    normed_anisotropy /= np.max(normed_anisotropy)

    fa = np.sum(contrasts.fa_map(ad_range, rd_range) * fractions, axis=(-1,-2))

    save_mrtrix(os.path.join(args.output_dir, 'intra_axonal_contrast.mif'), Image(intra_ax_contrast, vox=vox, comments=['Intra-axonal contrast']))
    save_mrtrix(os.path.join(args.output_dir, 'extra_axonal_contrast.mif'), Image(extra_ax_contrast, vox=vox, comments=['Extra-axonal contrast']))
    save_mrtrix(os.path.join(args.output_dir, 'free_water_contrast.mif'), Image(free_water_contrast, vox=vox, comments=['Free water contrast']))
    save_mrtrix(os.path.join(args.output_dir, 'fa.mif'), Image(fa, vox=vox, comments=['FA']))
if __name__ == '__main__':
    main()