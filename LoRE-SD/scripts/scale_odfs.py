# Provide functionality to scale the ODFs by the intra-axonal volume fractions
# or something similar.

# Add the source directory to the path
import sys
sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD/src')
sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD')

import numpy as np
import argparse
from mrtrix_io.io import load_mrtrix, save_mrtrix
from mrtrix_io.io.image import Image
import matplotlib.pyplot as plt
import seaborn as sns

def scale_odf_with_intra_axonal_vf(odf, intra_axonal_vf):
    """
    Scale the ODFs by the intra-axonal volume fractions.

    Parameters:
    - odf (numpy.ndarray): The ODFs to scale.
    - intra_axonal_vf (numpy.ndarray): The intra-axonal volume fractions.

    Returns:
    - numpy.ndarray: The scaled ODFs.
    """
    return np.nan_to_num(odf * intra_axonal_vf[...,None].astype(odf.dtype))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_odf', help='Input ODF file')
    parser.add_argument('CSD_ODF', help='CSD ODF')
    parser.add_argument('output_odf', help='Output ODF file')
    parser.add_argument('mask', help='Mask file')
    parser.add_argument('--contrast', help='Intra-axonal volume fraction', default=None)

    args = parser.parse_args()

    data = load_mrtrix(args.input_odf)
    mask = load_mrtrix(args.mask).data > .5
    odf = data.data
    vox=data.vox

    if args.contrast is not None:
        print(f'Scaling ODFs with {args.contrast}')
        contrast = load_mrtrix(args.contrast).data
        if contrast.ndim == 4:
            contrast = contrast[...,0]
        odf = scale_odf_with_intra_axonal_vf(odf, contrast)

    print('Scaling ODFs to match CSD ODF median')
    csd_odf = load_mrtrix(args.CSD_ODF).data

    actual_mean = np.nanmedian(odf[mask][...,0])
    required_mean = np.nanmedian(csd_odf[mask][...,0])

    odf = odf * (required_mean / actual_mean)

    save_mrtrix(args.output_odf, Image(odf, vox=data.vox, comments=[f'ODF scaled by {args.contrast} with same mean as {args.CSD_ODF}']))