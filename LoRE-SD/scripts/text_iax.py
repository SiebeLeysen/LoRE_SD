import argparse, os, sys
import numpy as np

sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD/src')
sys.path.append('/home/sleyse4/repos/LoRE_SD/LoRE-SD')

from src import contrasts

from mrtrix_io.io import save_mrtrix, load_mrtrix
from mrtrix_io.io.image import Image
import matplotlib.pyplot as plt
import seaborn as sns

def intra_axonal_contrast(ad_range, rd_range, with_isotropic=True, k=1, x0=0.5):
    """
    Calculate the intra-axonal contrast matrix.

    This function generates a matrix representing the contrast for intra-axonal spaces based on radial diffusivity (RD) ranges.
    The contrast is calculated using a sigmoid function applied to the RD values, normalized to the maximum value in the resulting matrix.

    Parameters:
    - ad_range (list or np.array): A list or array of AD values. Not directly used in calculations but required for consistent interface.
    - rd_range (list or np.array): A list or array of RD values.
    - with_isotropic (bool, optional): Whether to include isotropic components. Defaults to True.
    - k (float, optional): The steepness of the sigmoid function. Defaults to 1.
    - x0 (float, optional): The midpoint of the sigmoid function. Defaults to 0.5.

    Returns:
    - np.array: A 2D numpy array representing the normalized intra-axonal contrast.
    """
    out = np.zeros((len(ad_range), len(rd_range)))
    for i in range(len(rd_range)):
        if with_isotropic:
            out[i, :i + 1] = [1 - 1 / (1 + np.exp(-k * (j/(i+1) - x0))) for j in range(i+1)]
        else:
            out[i, :i] = [1 - 1 / (1 + np.exp(-k * (j/(i+1) - x0))) for j in range(i)]

    return out / np.max(out)

# Define decreasing functions
def sigmoid_function(j, k=1, x0=0.5):
    return 1 / (1 + np.exp(-k * (j - x0)))

def exponential_decay_function(j, rate=1):
    return np.exp(-rate * j)

def inverse_linear_function(j, scale=1):
    return 1 - scale * j

def inverse_logarithmic_function(j, base=np.e):
    return 1 - (np.log(j + 1) / np.log(base))

def inverse_tanh_function(j):
    return 1 - np.tanh(j)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Derive the intra-axonal contrast for multiple exponents')

    parser.add_argument('input_fractions', type=str, help='The gaussian fractions')
    
    parser.add_argument('output_dir', type=str, help='The output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ad, rd = np.linspace(0, 4e-3, 10), np.linspace(0, 4e-3, 10)

    f = load_mrtrix(args.input_fractions).data
    vox = load_mrtrix(args.input_fractions).vox

    for exp in range(1, 7):
        contrast = contrasts.get_contrast(f, ad, rd, contrasts.intra_axonal_contrast, True, exp)
        name = f'intra_axonal_contrast_exp_{exp}_iso.mif'
        save_mrtrix(os.path.join(args.output_dir, name), Image(contrast, vox=vox, comments=[f'Intra-axonal contrast with exp={exp}']))
    
    # # Define the number of rows and columns for the subplots
    # x_vals = np.linspace(0, .1, 3)
    # x_vals = np.array([0, .1])
    # k_vals = np.array([10, 30, 50])
    # num_rows = len(x_vals)
    # num_cols = len(k_vals)

    # # Create a figure with subplots
    # fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))

    # # Flatten the axes array for easy iteration
    # axes = axes.flatten()

    # # Loop through x and k values and plot heatmaps
    # plot_index = 0
    # for x in x_vals:
    #     for k in k_vals:
    #         sigmoid_contrast = contrasts.get_contrast(f, ad, rd, intra_axonal_contrast, False, k, x)
    #         matrix = intra_axonal_contrast(ad, rd, with_isotropic=True, k=k, x0=x)
    #         name = f'intra_axonal_contrast_x_{x}_iso_sigmoid_k_{k}.mif'
    #         save_mrtrix(os.path.join(args.output_dir, name), Image(sigmoid_contrast, vox=vox, comments=[f'Intra-axonal contrast with exxp={x} and k={k}']))

    #         sigmoid_contrast = contrasts.get_contrast(f, ad, rd, intra_axonal_contrast, True, k, x)
    #         matrix = intra_axonal_contrast(ad, rd, with_isotropic=True, k=k, x0=x)
    #         name = f'intra_axonal_contrast_x_{x}_iso_sigmoid_k_{k}_iso.mif'
    #         save_mrtrix(os.path.join(args.output_dir, name), Image(sigmoid_contrast, vox=vox, comments=[f'Intra-axonal contrast with exxp={x} and k={k}']))


    #         # Plot heatmap
    #         sns.heatmap(matrix, cmap='viridis', annot=True, fmt=".2f", ax=axes[plot_index], square=True)
    #         axes[plot_index].set_title(f'Heatmap (k={k}, x={x})')
    #         axes[plot_index].set_xlabel('RD Range')
    #         axes[plot_index].set_ylabel('AD Range')
    #         plot_index += 1
    # # Adjust layout
    # plt.tight_layout()

    # # Save the figure
    # plt.savefig(os.path.join(args.output_dir, 'all_heatmaps.png'), dpi=300)
    # plt.close()