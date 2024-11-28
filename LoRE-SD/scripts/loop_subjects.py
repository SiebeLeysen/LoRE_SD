import argparse
import subprocess
import os
import sys

sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD/src')
sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD')

from mrtrix_io.io import load_mrtrix, save_mrtrix
from mrtrix_io.io.image import Image

import numpy as np

def aic(k, target, pred, mask, subtract_mean=False):
    n = target.shape[-1]
    sigma_sq = RSS(target, pred, mask, subtract_mean)/n
    out = 2*k + n*np.log(sigma_sq)
    out[~mask] = 0
    return out

def RSS(target, pred, mask, subtract_mean=False):
    residual = target - pred
    if subtract_mean:
        residual -= np.mean(residual, axis=-1)

    out = np.sum(residual**2, axis=-1)

    out[~mask] = 0
    return out

if __name__ == '__main__':


    GLIOMA_DIR = '/DATASERVER/MIC/GENERAL/STAFF/sleyse4/u0152170/DATA/glioma'
    HEALTHY_DIR = '/DATASERVER/MIC/GENERAL/STAFF/sleyse4/u0152170/DATA'
    PREPROC_DIR = '/DATASERVER/MIC/GENERAL/STAFF/sleyse4/u0152170/Blind_Deconvolution/Preprocessing'
    # List all directories in DATA_DIR if 21 not in the directory name
    subjects = [x for x in os.listdir(GLIOMA_DIR) if '21' not in x]
    subjects += ['philips']
    # Loop through each subject
    for sub in subjects:
        print(f'Processing {sub}...')
        if sub == 'philips':
            os.makedirs(os.path.join(PREPROC_DIR, sub, 'grid5'), exist_ok=True)
            cmd = f'python /home/sleyse4/repos/LoRE_SD/LoRE-SD/dwi2decomposition.py ' \
                f'{os.path.join(HEALTHY_DIR, sub, "PREPROC/biascorr.mif")} {os.path.join(PREPROC_DIR, sub, "grid5")} ' \
                f'--mask {os.path.join(HEALTHY_DIR, sub, "PREPROC/mask.mif")} ' \
                f'--grid_size 20 --eval_matrix /LOCALDATA/sleyse4/Q_odf.npy --cores 150'
        else:
            os.makedirs(os.path.join(PREPROC_DIR, sub, 'grid5'), exist_ok=True)
            cmd = f'python /home/sleyse4/repos/LoRE_SD/LoRE-SD/dwi2decomposition.py ' \
                f'{os.path.join(GLIOMA_DIR, sub, "PREPROC/biascorr.mif")} {os.path.join(PREPROC_DIR, sub, "grid5")} ' \
                f'--mask {os.path.join(GLIOMA_DIR, sub, "PREPROC/mask.mif")} ' \
                f'--grid_size 20 --eval_matrix /LOCALDATA/sleyse4/Q_odf.npy --cores 150'
        subprocess.run(cmd, shell=True)
            