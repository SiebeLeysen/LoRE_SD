# Provide functionality to scale the ODFs by the intra-axonal volume fractions
# or something similar.

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

def plot_distribution(odfs):
    """
    Plot the distribution of the ODFs.

    Parameters:
    - odfs (numpy.ndarray): The ODFs to plot.
    """
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])


    sns.histplot(odfs[0][...,0].flatten(), ax=ax0)
    sns.histplot(odfs[1][...,0].flatten(), ax=ax1)
    sns.histplot(odfs[2][...,0].flatten(), ax=ax2)
    sns.histplot(odfs[3][...,0].flatten(), ax=ax3)
    plt.savefig('/LOCALDATA/sleyse4/LoRE_SD_tests/philips/fig.png', dpi=300)
    print('Saved figure')
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_odf', help='Input ODF file')
    parser.add_argument('CSD_ODF', help='CSD ODF')
    parser.add_argument('output_odf', help='Output ODF file')
    parser.add_argument('mask', help='Mask file')
    parser.add_argument('--intra_axonal_vf', help='Intra-axonal volume fraction', default=None)

    args = parser.parse_args()

    data = load_mrtrix(args.input_odf)
    mask = load_mrtrix(args.mask).data > .5
    odf = data.data
    odf_mask = odf[mask]
    odf_out = np.zeros_like(data.data)
    vox=data.vox
    if args.intra_axonal_vf is not None:
        print('Scaling ODFs with intra-axonal volume fractions')
        intra_axonal_vf = load_mrtrix(args.intra_axonal_vf).data
        odf = scale_odf_with_intra_axonal_vf(odf, intra_axonal_vf)
        odf_mask = scale_odf_with_intra_axonal_vf(odf_mask, intra_axonal_vf[mask])

    if args.CSD_ODF is not None:
        print('Scaling ODFs to match CSD ODF mean')
        csd_odf = load_mrtrix(args.CSD_ODF).data
        csd_odf_mask = csd_odf[mask]

    actual_mean = np.nanmean(odf_mask[...,0])
    required_mean = np.nanmean(csd_odf_mask[...,0])
    odf_mask = odf_mask * (required_mean / actual_mean)

    odf_out[mask] = odf_mask
    save_mrtrix(args.output_odf, Image(odf_out, vox=data.vox, comments=[f'ODF scaled by {args.intra_axonal_vf} with same mean as {args.CSD_ODF}']))