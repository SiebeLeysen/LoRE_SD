import argparse
import numpy as np

from mrtrix_io.io import load_mrtrix, save_mrtrix
from mrtrix_io.io.image import Image


def add_noise(dwi_in, tissue_seg, dwi_out, snr):
    print(f'Loading data from {dwi_in}')
    data = load_mrtrix(dwi_in)
    dwi = data.data
    grad = data.grad
    vox = data.vox
    print(f'Loading data from {tissue_seg}')
    wm_mask = load_mrtrix(tissue_seg).data[...,2] > .5
    b0 = dwi[...,np.round(grad[:,-1],-2)==0]
    avg_b0 = np.mean(b0[wm_mask])

    std_noise = avg_b0 / snr

    print(f'Adding noise with std={std_noise}')
    noise1 = np.random.normal(0, std_noise, size=dwi.shape)
    noise2 = np.random.normal(0, std_noise, size=dwi.shape)
    print(f'Adding Rician noise')
    noisy_dwi = np.sqrt((dwi + noise1)**2 + noise2**2)

    print(f'Saving data to {dwi_out}')
    
    save_mrtrix(dwi_out, Image(noisy_dwi, grad=grad, vox=vox, comments=f'Noise added with SNR={snr}'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Add Rician noise to a DWI image w.r.t. mean b0 intensity in white matter.')
    parser.add_argument('noise_free', help='Input noise_free image')
    parser.add_argument('tissue_seg', help='5tt segmentation')
    parser.add_argument('output', help='Output image')
    parser.add_argument('snr', type=float, help='Signal-to-noise ratio')

    args = parser.parse_args()

    add_noise(args.noise_free, args.tissue_seg, args.output, args.snr)