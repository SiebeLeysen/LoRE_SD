import argparse
import numpy as np

from mrtrix_io.io import load_mrtrix, save_mrtrix
from mrtrix_io.io.image import Image

def aic(k, target, pred, mask, subtract_mean=False):
    n = target.shape[-1]
    sigma_sq = RSS(target, pred, mask, subtract_mean) / n
    out = 2 * k + n * np.log(sigma_sq)
    out[~mask] = 0
    return out


def RSS(target, pred, mask, subtract_mean=False):
    residual = target - pred
    if subtract_mean:
        residual -= np.mean(residual, axis=-1)

    out = np.sum(residual ** 2, axis=-1)

    out[~mask] = 0
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target', help='Target data')
    parser.add_argument('pred', help='Predicted data')
    parser.add_argument('output', help='Output image')
    parser.add_argument('mask', help='Mask')
    parser.add_argument('parameter_count', type=int, help='Number of parameters in the model')
    parser.add_argument('--subtract_mean', type=int, default=0, help='Subtract mean from residuals')
    args = parser.parse_args()

    target = load_mrtrix(args.target)
    pred = load_mrtrix(args.pred)
    mask = load_mrtrix(args.mask)
    k = args.parameter_count

    out = aic(k, target.data, pred.data, mask.data, args.subtract_mean == 1)
    save_mrtrix(args.output, Image(out, vox=target.vox))