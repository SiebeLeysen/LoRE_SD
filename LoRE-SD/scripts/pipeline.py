import argparse
import numpy as np
import os
import sys
import subprocess
from scipy.ndimage import binary_erosion

sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD/src')
sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD')

from src import contrasts
from mrtrix_io.io import load_mrtrix, save_mrtrix
from mrtrix_io.io.image import Image
from src.utils import math_utils

def erode_mask(mask, k=2):
    mask_data = load_mrtrix(mask)
    eroded = binary_erosion(mask_data.data > .5, structure=np.ones((k,k,k))).astype(mask_data.data.dtype)
    save_mrtrix(mask, Image(eroded, vox=mask_data.vox, comments='Eroded mask'))


def dwi2mask(dwi_dir):
    print(f'Running dwi2mask ...')

    cmd_mean_b0 = f'dwiextract {os.path.join(dwi_dir, "biascorr.mif")} -bzero -  -quiet -force| mrmath - mean -axis 3 {os.path.join(dwi_dir, "b0.nii.gz")}  -quiet -force'
    subprocess.run(cmd_mean_b0, shell=True)
    cmd_bet = f'bet {os.path.join(dwi_dir, "b0.nii.gz")} {os.path.join(dwi_dir, "tmp.nii.gz")} -m -R -f .3'
    subprocess.run(cmd_bet, shell=True)
    cmd_convert = f'mrconvert {os.path.join(dwi_dir, "tmp_mask.nii.gz")} {os.path.join(dwi_dir, "mask.mif")} -force'
    subprocess.run(cmd_convert, shell=True)
    cmd_rm = f'rm {os.path.join(dwi_dir, "tmp.nii.gz")} {os.path.join(dwi_dir, "tmp_mask.nii.gz")}'
    subprocess.run(cmd_rm, shell=True)

    erode_mask(os.path.join(dwi_dir, 'mask.mif'))

def tissue_segmentation(T1_dir, output_dir):
    # If tissue segmentation is already done, skip
    if os.path.exists(os.path.join(output_dir, '5tt.mif')):
        return
    cmd_5tt = f'5ttgen fsl {os.path.join(T1_dir, "T1w.nii.gz")} {os.path.join(output_dir, "5tt.mif")} \
        -nocrop -force'
    print(f'Running Tissue Segmentation ...')
    subprocess.run(cmd_5tt, shell=True)

def coregister_t1w(dwi_dir, t1w_dir, output_dir, lesion_dir=None):
    # If registration is already done, skip
    if os.path.exists(os.path.join(output_dir, '5tt_in_dwi.mif')):
        return
    print(f'Running T1-DWI Coregistration ...')

    # First extract the brain from T1
    cmd_bet_t1 = f'bet {os.path.join(t1w_dir, "T1w.nii.gz")} {os.path.join(output_dir, "T1w_bet.nii.gz")} -R -f .5'
    subprocess.run(cmd_bet_t1, shell=True)

    cmd_ants_register = f'antsRegistrationSyN.sh -d 3 -f {os.path.join(dwi_dir, "b0.nii.gz")} ' \
        f'-m {os.path.join(output_dir, "T1w_bet.nii.gz")} -n 100 -t a -o {os.path.join(output_dir, "t1w_to_dwi_")}'
    subprocess.run(cmd_ants_register, shell=True)

    # # Remove the T1w_bet.nii.gz
    # cmd_rm_bet = f'rm {os.path.join(output_dir, "T1w_bet.nii.gz")}'
    # subprocess.run(cmd_rm_bet, shell=True)

    cmd_convert_transform_to_txt = f'ConvertTransformFile 3 {os.path.join(output_dir, "t1w_to_dwi_0GenericAffine.mat")} ' \
        f'{os.path.join(output_dir, "t1w_to_dwi_0GenericAffine.txt")}'
    subprocess.run(cmd_convert_transform_to_txt, shell=True)

    cmd_transform_to_mrtrix = f'transformconvert {os.path.join(output_dir, "t1w_to_dwi_0GenericAffine.txt")} ' \
        f'itk_import {os.path.join(output_dir, "t1w_to_dwi_mrtrix.txt")} -force'
    subprocess.run(cmd_transform_to_mrtrix, shell=True)

    cmd_t1_to_dwi = f'mrtransform {os.path.join(t1w_dir, "T1w.nii.gz")} -linear {os.path.join(output_dir, "t1w_to_dwi_mrtrix.txt")} ' \
        f'{os.path.join(output_dir, "T1w_in_dwi.mif")} -force'
    subprocess.run(cmd_t1_to_dwi, shell=True)

    cmd_5tt_to_dwi = f'mrtransform {os.path.join(output_dir, "5tt.mif")} -linear {os.path.join(output_dir, "t1w_to_dwi_mrtrix.txt")} ' \
        f'{os.path.join(output_dir, "5tt_in_dwi.mif")} -force'
    subprocess.run(cmd_5tt_to_dwi, shell=True)

    if lesion_dir is not None:
        cmd_lesion_to_dwi = f'mrtransform {os.path.join(lesion_dir, "lesion.nii.gz")} -linear {os.path.join(output_dir, "t1w_to_dwi_mrtrix.txt")} ' \
            f'{os.path.join(output_dir, "lesion_in_dwi.mif")} -force'
        subprocess.run(cmd_lesion_to_dwi, shell=True)

        cmd_lesion_out_of_5tt = f'mrcalc 1 {os.path.join(output_dir, "lesion_in_dwi.mif")} -subtract - | mrcalc - .5 -gt - | ' \
        f' mrcalc - {os.path.join(output_dir, "5tt_in_dwi.mif")} -mult - | mrconvert - {os.path.join(output_dir, "5tt_without_lesion.mif")} -coord 3 0:1:3 -quiet -force'
        subprocess.run(cmd_lesion_out_of_5tt, shell=True)

        cmd_add_lesion_last = f'mrcat {os.path.join(output_dir, "5tt_without_lesion.mif")} {os.path.join(output_dir, "lesion_in_dwi.mif")} {os.path.join(output_dir, "5tt_lesion_separate.mif")} -axis 3 -force'
        subprocess.run(cmd_add_lesion_last, shell=True)

def lesion2edge(output_dir):
    if os.path.exists(os.path.join(output_dir, 'lesion_edge.mif')):
        return
    # Also regrid the lesion to a much finer grid, then detects its edges and finally smooth it a bit.
    cmd_to_edge = f'python /home/sleyse4/repos/LoRE_SD/LoRE-SD/scripts/mask2contour.py {os.path.join(output_dir, "lesion_in_dwi.mif")} {os.path.join(output_dir, "lesion_edge.mif")}'
    subprocess.run(cmd_to_edge, shell=True)

    
def dwi2mtcsd(dwi_dir, output_dir, lesion_dir=None):
    print(f'Running MSMT-CSD ...')
    os.makedirs(os.path.join(output_dir, 'MTCSD'), exist_ok=True)
    cmd_response = f'dwi2response dhollander {os.path.join(dwi_dir, "biascorr.mif")} {os.path.join(output_dir, "MTCSD", "wm.txt")} ' \
        f'{os.path.join(output_dir, "MTCSD", "gm.txt")} {os.path.join(output_dir, "MTCSD", "csf.txt")} -mask {os.path.join(dwi_dir, "mask.mif")} -force ' \
        f'-voxels {os.path.join(output_dir, "MTCSD", "response_voxels.mif")}'
    subprocess.run(cmd_response, shell=True)
    cmd_fod = f'dwi2fod msmt_csd {os.path.join(dwi_dir, "biascorr.mif")} {os.path.join(output_dir, "MTCSD", "wm.txt")} ' \
        f'{os.path.join(output_dir, "MTCSD", "wm.mif")} {os.path.join(output_dir, "MTCSD", "gm.txt")} {os.path.join(output_dir, "MTCSD", "gm.mif")} ' \
        f'{os.path.join(output_dir, "MTCSD", "csf.txt")} {os.path.join(output_dir, "MTCSD", "csf.mif")} -predicted_signal ' \
        f'{os.path.join(output_dir, "MTCSD", "predicted_signal.mif")} -mask {os.path.join(dwi_dir, "mask.mif")} -force'
    subprocess.run(cmd_fod, shell=True)
    pred = load_mrtrix(os.path.join(output_dir, 'MTCSD', 'predicted_signal.mif')).data
    mask = load_mrtrix(os.path.join(dwi_dir, 'mask.mif')).data > .5
    rmse = math_utils.rmse(pred, load_mrtrix(os.path.join(dwi_dir, 'biascorr.mif')).data, mask)
    save_mrtrix(os.path.join(output_dir, 'MTCSD', 'rmse.mif'), Image(rmse, vox=load_mrtrix(os.path.join(dwi_dir, 'biascorr.mif')).vox, comments='Root Mean Squared Error by MSMT-CSD'))

def dwi2lore(dwi_dir, output_dir, cores):
    print(f'Running LoRE ...')
    os.makedirs(os.path.join(output_dir, 'LoRE'), exist_ok=True)
    cmd_lore = f'python /home/sleyse4/repos/LoRE_SD/LoRE-SD/dwi2decomposition.py ' \
        f'{os.path.join(dwi_dir, "biascorr.mif")} {os.path.join(output_dir, "LoRE")} ' \
        f'--cores {cores} --mask {os.path.join(dwi_dir, "mask.mif")}'
    subprocess.run(cmd_lore, shell=True)

def tissue_seg_2_wm_mask(output_dir):
    if os.path.exists(os.path.join(output_dir, 'wm_mask.mif')):
        return
    cmd_regrid_5tt = f'mrgrid {os.path.join(output_dir, "5tt.mif")} regrid -template {os.path.join(output_dir, "MTCSD", "wm.mif")} ' \
        f'{os.path.join(output_dir, "5tt_regrid.mif")} -interp nearest -force -quiet'

    subprocess.run(cmd_regrid_5tt, shell=True)

    cmd_to_mask = f'mrconvert {os.path.join(output_dir, "5tt_regrid.mif")} -coord 3 2 -axes 0,1,2 - | mrcalc - .99 -ge {os.path.join(output_dir, "wm_mask.mif")} -force'
    subprocess.run(cmd_to_mask, shell=True)

def lore2contrasts(dwi_dir, output_dir):
    print(f'Running LoRE to Contrasts ...')
    cmd_contrasts = f'python /home/sleyse4/repos/LoRE_SD/LoRE-SD/scripts/decomposition2contrast.py ' \
        f'{os.path.join(output_dir, "LoRE", "gaussian_fractions.mif")} {os.path.join(output_dir, "LoRE", "response.mif")} ' \
        f'{os.path.join(output_dir, "LoRE")}'
    subprocess.run(cmd_contrasts, shell=True)

    cmd_scale_odf = f'python /home/sleyse4/repos/LoRE_SD/LoRE-SD/scripts/scale_odfs.py ' \
        f'{os.path.join(output_dir, "LoRE", "odf.mif")} {os.path.join(output_dir, "MTCSD", "wm.mif")} ' \
        f'{os.path.join(output_dir, "LoRE", "odf_anisotropy.mif")} {os.path.join(output_dir, "wm_mask.mif")} ' \
        f'--contrast {os.path.join(output_dir, "LoRE", "anisotropy.mif")}'
    subprocess.run(cmd_scale_odf, shell=True)
    cmd_scale_odf = f'python /home/sleyse4/repos/LoRE_SD/LoRE-SD/scripts/scale_odfs.py ' \
        f'{os.path.join(output_dir, "LoRE", "odf.mif")} {os.path.join(output_dir, "MTCSD", "wm.mif")} ' \
        f'{os.path.join(output_dir, "LoRE", "odf_fa.mif")} {os.path.join(output_dir, "wm_mask.mif")} ' \
        f'--contrast {os.path.join(output_dir, "LoRE", "fa.mif")}'
    subprocess.run(cmd_scale_odf, shell=True)
    cmd_scale_odf = f'python /home/sleyse4/repos/LoRE_SD/LoRE-SD/scripts/scale_odfs.py ' \
        f'{os.path.join(output_dir, "LoRE", "odf.mif")} {os.path.join(output_dir, "MTCSD", "wm.mif")} ' \
        f'{os.path.join(output_dir, "LoRE", "odf_normed.mif")} {os.path.join(output_dir, "wm_mask.mif")} ' \
        f'--contrast {os.path.join(output_dir, "LoRE", "normed_anisotropy.mif")}'
    subprocess.run(cmd_scale_odf, shell=True)
    cmd_scale_odf = f'python /home/sleyse4/repos/LoRE_SD/LoRE-SD/scripts/scale_odfs.py ' \
        f'{os.path.join(output_dir, "LoRE", "odf.mif")} {os.path.join(output_dir, "MTCSD", "wm.mif")} ' \
        f'{os.path.join(output_dir, "LoRE", "odf_iax.mif")} {os.path.join(output_dir, "wm_mask.mif")} ' \
        f'--contrast {os.path.join(output_dir, "LoRE", "intra_axonal_contrast.mif")}'
    subprocess.run(cmd_scale_odf, shell=True)
    cmd_scale_odf = f'python /home/sleyse4/repos/LoRE_SD/LoRE-SD/scripts/scale_odfs.py ' \
        f'{os.path.join(output_dir, "LoRE", "odf.mif")} {os.path.join(output_dir, "MTCSD", "wm.mif")} ' \
        f'{os.path.join(output_dir, "LoRE", "odf_wm.mif")} {os.path.join(output_dir, "wm_mask.mif")} ' \
        f'--contrast {os.path.join(output_dir, "MTCSD", "wm.mif")}'
    subprocess.run(cmd_scale_odf, shell=True)
    cmd_scale_odf = f'python /home/sleyse4/repos/LoRE_SD/LoRE-SD/scripts/scale_odfs.py ' \
        f'{os.path.join(output_dir, "LoRE", "odf.mif")} {os.path.join(output_dir, "MTCSD", "wm.mif")} ' \
        f'{os.path.join(output_dir, "LoRE", "odf_new.mif")} {os.path.join(output_dir, "wm_mask.mif")} ' \
        f'--contrast {os.path.join(output_dir, "LoRE", "new_contrast.mif")}'
    subprocess.run(cmd_scale_odf, shell=True)



def angular_corr(output_dir):
    print(f'Running Angular Correlation ...')
    cmd_angular_corr = f'python /home/sleyse4/repos/LoRE_SD/LoRE-SD/scripts/angular_correlation.py ' \
        f'{os.path.join(output_dir, "LoRE", "odf.mif")} {os.path.join(output_dir, "MTCSD", "wm.mif")} ' \
        f'{os.path.join(output_dir, "acc.mif")}'
    subprocess.run(cmd_angular_corr, shell=True)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_dwi_dir', type=str, help='Path to the directory containing the preprocessed DWI data')
    parser.add_argument('input_t1w_dir', type=str, help='Path to the directory containing the preprocessed T1w data')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    parser.add_argument('--lesion_dir', type=str, help='Path to the lesion mask directory', default=None)
    parser.add_argument('--cores', type=int, help='Number of cores to use', default=1)

    args = parser.parse_args()

    dwi2mask(args.input_dwi_dir)
    tissue_segmentation(args.input_t1w_dir, args.output_dir)
    coregister_t1w(args.input_dwi_dir, args.input_t1w_dir, args.output_dir, args.lesion_dir)
    if args.lesion_dir is not None:
        lesion2edge(args.output_dir)
    dwi2mtcsd(args.input_dwi_dir, args.output_dir)
    dwi2lore(args.input_dwi_dir, args.output_dir, args.cores)
    tissue_seg_2_wm_mask(args.output_dir)
    lore2contrasts(args.input_dwi_dir, args.output_dir)
    angular_corr(args.output_dir)


