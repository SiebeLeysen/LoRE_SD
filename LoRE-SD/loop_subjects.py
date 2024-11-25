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


    DATA_DIR = '/DATASERVER/MIC/GENERAL/STAFF/sleyse4/u0152170/DATA/'
    PREPROC_DIR = '/DATASERVER/MIC/GENERAL/STAFF/sleyse4/u0152170/Blind_Deconvolution/Preprocessing'
    # List all directories in DATA_DIR if 21 not in the directory name
    subjects = [x for x in os.listdir(DATA_DIR) if '21' not in x]
    subjects = ['philips']
    # Loop through each subject
    for sub in subjects:
        print(f'Processing {sub}...')
        # Calculate the ACC
        
        gt = load_mrtrix(os.path.join(DATA_DIR, sub, "PREPROC/biascorr.mif"))
        gt_dwi = gt.data
        grad = gt.grad
        vox = gt.vox
        mask = load_mrtrix(os.path.join(DATA_DIR, sub, "PREPROC/mask.mif")).data > 0.5

        pred_lore = load_mrtrix(os.path.join(PREPROC_DIR, sub, "LoRE/reconstructed.mif")).data
        pred_mtcsd = load_mrtrix(os.path.join(PREPROC_DIR, sub, "MTCSD/predicted_signal.mif")).data

        aic_lore = aic(98, 
                       gt_dwi, 
                       pred_lore,
                       mask)
        
        save_mrtrix(os.path.join(PREPROC_DIR, sub, "LoRE/aic.mif"), Image(aic_lore, vox=vox, comments='AIC by LoRE-SD'))

        aic_mtcsd = aic(47,
                        gt_dwi, 
                        pred_mtcsd,
                        mask)
        
        save_mrtrix(os.path.join(PREPROC_DIR, sub, "MTCSD/aic.mif"), Image(aic_mtcsd, vox=vox, comments='AIC by MTCSD'))
        