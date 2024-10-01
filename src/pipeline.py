import subprocess, os, argparse

def copy_data(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)

    subprocess.run(['cp', src, dst])

def dwi2mask(dwi, mask):
    avg_b0_cmd = f'dwiextract {dwi} - -bzero | mrmath - mean b0.nii.gz -axis 3'
    subprocess.run(avg_b0_cmd, shell=True)
    bet_cmd = f'bet b0.nii.gz mask.nii.gz -R -m -f 0.25'
    subprocess.run(bet_cmd, shell=True)
    cvt_cmd = f'mrconvert mask_mask.nii.gz {mask}'
    subprocess.run(cvt_cmd, shell=True)
    rm_cmd = 'rm b0.nii.gz mask.nii.gz mask_mask.nii.gz'
    subprocess.run(rm_cmd, shell=True)

def create_subdirs(root_dir):
    try:
        os.makedirs(os.path.join(root_dir, 'LORE'))
        os.makedirs(os.path.join(root_dir, 'MTCSD'))
    except FileExistsError:
        pass

def dwi2response(dwi, mask, save_dir):
    cmd = f'dwi2response dhollander {dwi} {save_dir}/wm.txt {save_dir}/gm.txt {save_dir}/csf.txt -mask {mask}'
    subprocess.run(cmd, shell=True)

def dwi2fod(dwi, mask, save_dir):
    cmd = f'dwi2fod msmt_csd {dwi} {save_dir}/wm.txt {save_dir}/wm.mif {save_dir}/gm.txt {save_dir}/gm.mif {save_dir}/csf.txt {save_dir}/csf.mif -mask {mask} -predicted_signal {save_dir}/recon.mif'
    subprocess.run(cmd, shell=True)
    rmse_cmd = f'python /home/sleyse4/repos/LoRE_SD/src/RMSE.py {dwi} {save_dir}/recon.mif {save_dir}/rmse.mif --mask {mask}'
    subprocess.run(rmse_cmd, shell=True)

def dwi2LoRE(dwi, mask, save_dir):
    dwi2decomposition_cmd = f'python /home/sleyse4/repos/LoRE_SD/src/dwi2decomposition.py {dwi} {save_dir} --mask {mask} --cores 50'
    subprocess.run(dwi2decomposition_cmd, shell=True)
    decomposition2contrast_cmd = f'python /home/sleyse4/repos/LoRE_SD/src/decomposition2contrast.py {save_dir}/gaussian_fractions.mif {save_dir} --with_isotropic 1'
    subprocess.run(decomposition2contrast_cmd, shell=True)

def scale_odfs_to_mtcsd(lore_dir, csd_dir, mask):
    scle_cmd = f'python /home/sleyse4/repos/LoRE_SD/src/scale_odfs.py {lore_dir}/odf.mif {csd_dir}/wm.mif {lore_dir}/odf_scaled.mif {mask} --intra_axonal_vf {lore_dir}/intra_axonal_contrast.mif'
    subprocess.run(scle_cmd, shell=True)

def to_acc_maps(lore_dir, csd_dir, save_dir):
    acc_cmd = f'python /home/sleyse4/repos/LoRE_SD/src/angular_correlation.py {lore_dir}/odf.mif {csd_dir}/wm.mif {save_dir}/acc.mif'
    subprocess.run(acc_cmd, shell=True)

def to_aic_maps(lore_dir, csd_dir, save_dir):
    k_lore = 99
    k_csd = 47
    aic_lore = f'python /home/sleyse4/repos/LoRE_SD/src/get_aic.py {save_dir}/dwi.mif {lore_dir}/reconstructed.mif {lore_dir}/aic.mif {save_dir}/mask.mif {k_lore}'
    aic_csd = f'python /home/sleyse4/repos/LoRE_SD/src/get_aic.py {save_dir}/dwi.mif {csd_dir}/recon.mif {csd_dir}/aic.mif {save_dir}/mask.mif {k_csd}'
    subprocess.run(aic_lore, shell=True)
    subprocess.run(aic_csd, shell=True)

def run_pipeline(src, dst):
    copy_data(src, dst)
    if not os.path.exists(f'{dst}/mask.mif'):
        dwi2mask(f'{dst}/dwi.mif', f'{dst}/mask.mif')
    else:
        print('Mask already exists')
        
    create_subdirs(dst)

    if not os.path.exists(f'{dst}/MTCSD/wm.txt'):
        dwi2response(f'{dst}/dwi.mif', f'{dst}/mask.mif', f'{dst}/MTCSD')
    else:
        print('Response functions already exist')

    if not os.path.exists(f'{dst}/MTCSD/wm.mif'):
        dwi2fod(f'{dst}/dwi.mif', f'{dst}/mask.mif', f'{dst}/MTCSD')
    else:
        print('FODs already exist')

    if not os.path.exists(f'{dst}/LORE/odf.mif'):
        dwi2LoRE(f'{dst}/dwi.mif', f'{dst}/mask.mif', f'{dst}/LORE')
    else:
        print('LoRE decomposition already exists')

    if not os.path.exists(f'{dst}/LORE/odf_scaled.mif'):
        scale_odfs_to_mtcsd(f'{dst}/LORE', f'{dst}/MTCSD', f'{dst}/mask.mif')
    else:
        print('Scaled ODFs already exist')

    if not os.path.exists(f'{dst}/LORE/acc.mif'):
        to_acc_maps(f'{dst}/LORE', f'{dst}/MTCSD', f'{dst}')
    else:
        print('Angular correlation maps already exist')
    
    if not os.path.exists(f'{dst}/LORE/aic.mif'):
        to_aic_maps(f'{dst}/LORE', f'{dst}/MTCSD', f'{dst}')
    else:
        print('AIC maps already exist')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Decompose the dMRI data into voxel-level ODFs and response functions using LoRE-SD.')
    parser.add_argument('input', help='Input image')
    parser.add_argument('output_dir', help='Output directory')

    args = parser.parse_args()

    run_pipeline(args.input, args.output_dir)