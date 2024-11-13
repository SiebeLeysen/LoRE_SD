import argparse
import numpy as np
import os
import sys
import subprocess

sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD/src')
sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD')

from src import contrasts
from src.optimisation import optimise
from src.utils import SphericalHarmonics as sh

from mrtrix_io.io import load_mrtrix, save_mrtrix
from mrtrix_io.io.image import Image

def handle_input(args):
    """
    Handles the input image file based on its format.

    If the input image is not in '.mif' format, it converts the image to '.mif' using the provided bvecs and bvals files.
    Otherwise, it directly loads the '.mif' file.

    Parameters:
    - args (Namespace): Command line arguments containing the input image path, bvecs, and bvals file paths.

    Returns:
    - Image: The loaded or converted MRtrix image object.

    Raises:
    - AssertionError: If the input image is not in '.mif' format and either bvecs or bvals files are not provided.
    """
    if not args.input.endswith('.mif'):
        assert args.bvecs is not None and args.bvals is not None, 'bvecs and bvals files are required for conversion.'
        return convert_to_mif(args.input, args.bvecs, args.bvals)
    else:
        return load_mrtrix(args.input)

def adjust_cores(requested_cores):
    """
    Adjusts the number of cores to use based on the system's capabilities and the user's request.

    Parameters:
    - requested_cores (int): The number of cores requested by the user.

    Returns:
    - int: The actual number of cores to use, which is the minimum of the requested cores and the system's available cores.
    """
    max_cores = os.cpu_count()
    return min(requested_cores, max_cores)

def prepare_parameters(grid_size):
    """
    Prepares the parameters for the signal decomposition process.

    Generates linearly spaced values for Axial Diffusivity (AD) and Radial Diffusivity (RD) based on the specified grid size.

    Parameters:
    - grid_size (int): The number of linearly spaced values to generate for AD and RD.

    Returns:
    - tuple: A tuple containing two numpy arrays for AD and RD values.
    """
    ad_list = np.linspace(0, 4e-3, grid_size)
    rd_list = np.linspace(0, 4e-3, grid_size)
    return ad_list, rd_list

def save_outputs(args, odf, responses, gaussian_fractions, rec, rmse, vox, grad):
    save_mrtrix(os.path.join(args.output_dir, 'odf.mif'), Image(odf, vox=vox, comments='ODF estimations by LoRE-SD'))
    save_mrtrix(os.path.join(args.output_dir, 'response.mif'), Image(responses, vox=vox, comments='Response function estimations by LoRE-SD'))
    save_mrtrix(os.path.join(args.output_dir, 'gaussian_fractions.mif'), Image(gaussian_fractions, vox=vox, comments='Gaussian fractions estimations by LoRE-SD'))
    save_mrtrix(os.path.join(args.output_dir, 'reconstructed.mif'), Image(rec, grad=grad, vox=vox, comments='Reconstructed signal by LoRE-SD'))
    save_mrtrix(os.path.join(args.output_dir, 'rmse.mif'), Image(rmse, vox=vox, comments='Root Mean Squared Error by LoRE-SD'))

    save_mrtrix(os.path.join(args.output_dir, 'outer_rf.mif'), Image(optimise.expand_response(responses)[...,-1,:], vox=vox, comments='Outer response function by LoRE-SD'))

def convert_to_mif(nii_path, bvecs_path, bvals_path):
    cmd_nii_to_img = f'mrconvert {nii_path} {nii_path.replace(".nii.gz", ".mif")} -fslgrad {bvecs_path} {bvals_path} -quiet'
    subprocess.run(cmd_nii_to_img.split(' '))
    out = load_mrtrix(nii_path.replace('.nii.gz', '.mif'))
    subprocess.run(f'rm {nii_path.replace(".nii.gz", ".mif")}'.split(' '))
    return out

def get_mask(input_image, cores):
    subprocess.run(['dwi2mask', input_image, 'tmp_mask.mif', '-nthreads', str(cores)])
    mask = load_mrtrix('tmp_mask.mif').data > .5
    subprocess.run(['rm', 'tmp_mask.mif'])
    return mask

def main():
    parser = argparse.ArgumentParser(
        description='Decompose the dMRI data into voxel-level ODFs and response functions using LoRE-SD.')
    parser.add_argument('input', help='Input image')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--reg', help='Regularisation parameter', type=float, default=1e-3)
    parser.add_argument('--grid_size', help='Grid size (Number of linearly spaced values for AD and RD)', type=int,
                        default=10)
    parser.add_argument('--cores', help='Number of cores to use', type=int, default=1)
    parser.add_argument('--bvecs', help='Path to the bvecs file', default=None)
    parser.add_argument('--bvals', help='Path to the bvals file', default=None)
    parser.add_argument('--mask', help='Path to the mask file', default=None)
    parser.add_argument('--high_res_data', help='Path to the high resolution data', default=None)

    args = parser.parse_args()

    input_args = handle_input(args)
    cores = adjust_cores(args.cores)
    if args.mask is not None:
        mask = load_mrtrix(args.mask).data > .5
    else:
        mask = get_mask(args.input, cores)
    ad_list, rd_list = prepare_parameters(args.grid_size)

    out = optimise.get_signal_decomposition(input_args.data, mask, input_args.grad, ad_list, rd_list, args.reg, cores=cores)

    vox = load_mrtrix(args.input).vox

    save_outputs(args, out['odf'], out['response'], out['gaussian_fractions'], out['reconstructed'], out['rmse']*mask, vox, input_args.grad)

    if args.high_res_data is not None:
        high_res_data = load_mrtrix(args.high_res_data)
        recon_hr = sh.calcdwi(sh.sphconv(out['response'], out['odf']), high_res_data.grad)
        rmse_hr = np.linalg.norm(high_res_data.data - recon_hr, axis=-1) / np.sqrt(high_res_data.data.shape[-1])
        save_mrtrix(os.path.join(args.output_dir, 'reconstructed_hr.mif'), Image(recon_hr, vox=high_res_data.vox, grad=high_res_data.grad, comments='Reconstructed signal by LoRE-SD'))
        save_mrtrix(os.path.join(args.output_dir, 'rmse_hr.mif'), Image(rmse_hr, vox=high_res_data.vox, comments='Root Mean Squared Error by LoRE-SD'))

if __name__ == '__main__':
    main()
