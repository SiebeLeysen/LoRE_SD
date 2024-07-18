import argparse

import numpy as np
import os
import subprocess

def rmse(dwi1, dwi2, mask):
    """
    Root Mean Squared Error of 2 DWI images. RMSE is calculated on the voxel level.
    The number of samples thus equals the number of gradient directions.
    :param dwi1: First DWI signal
    :param dwi2: Second DWI signal, shape must match dwi1.
    :param mask: Binary brain mask. RMSE is only calculated for brain voxels.
    :return: Array with per-voxel RMSE values.
    """
    assert dwi1.shape == dwi2.shape, f'Input shapes do not align. dwi1 has shape {dwi1.shape} while dwi2 has shape {dwi2.shape}'
    res = np.zeros(dwi1.shape[:-1])
    res[mask] = np.linalg.norm(dwi1[mask] - dwi2[mask], ord=2, axis=-1) / np.sqrt(dwi1.shape[-1])
    return res

def save_vector(output_path, v):
    """
    Save a vector or matrix. This is mainly used to save individual ODFs or response functions.
    These can then be displayed in MRtrix using the `shview` command.
    :param output_path: Path to write text file to.
    :param v: Vector or matrix to write.
    :return: None
    """
    assert len(v.shape) <= 2
    with open(output_path, 'w') as f:
        if len(v.shape) > 1:
            for row in v:
                for elem in row:
                    f.write(str(elem) + ' ')
                f.write('\n')
        else:
            for elem in v:
                f.write(str(elem) + ' ')
        f.close()
        
def load_vector(input_path):
    """
    Read a vector or matrix. This allows reading response functions and ODFs from MRtrix.
    :param input_path: Input file path name.
    :return: Vector or matrix in stored in the file.
    """
    res = []
    with open(input_path) as f:
        line = f.readline()
        while line:
            if '#' not in line:
                line = line.strip('\n')
                line = line.split(' ')
                line = list(filter(lambda e: e != '', line))
                line = list(map(lambda s: float(s), line))
                res.append(line)
            line = f.readline()
    return np.array(res, dtype=np.float32)


def create_output_array(data, mask, output_shape):
    """
    Create an output array from flat input data. Initialise an empty array with shape output_shape.
    `data` is then written to those locations where the brain mask is non-zero.
    :param data: Input data with as many elements as there are True values in `mask`.
    :param mask: Binary brain mask.
    :param output_shape: Output shape of the resulting data.
    :return: Array with shape `output_shape` with voxels within the brain mask set to `data`.
    """
    output_arr = np.zeros(output_shape, dtype=data.dtype)
    output_arr[mask] = data
    return output_arr

def clean_axis(ax):
    """
    Remove all tick labels, ticks and spines from the axis to clean it up. Mainly used to show images with plt.imshow().
    :param ax: matplotlib.pyplot axis
    :return: None
    """
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def pretty_print_gradient_table(grad):
    """
    Print the gradient table as a list of b-values and number of directions for each b-value.
    :param grad: Gradient table
    :return: None
    """
    b_values = grad[:, -1]
    unique_b_values, counts = np.unique(np.round(b_values, -2), return_counts=True)
    b_value_counts = dict(zip(unique_b_values, counts))

    print('b-value   |   count   ')
    print('----------------------')

    for b in unique_b_values:
        print(f"{int(b):5d}     |   {b_value_counts[b]}")

def create_gradient_table(bvals, dirs_per_shell):
    """
    Create a gradient table with a specific number of directions per shell.

    Args:
        bvals (list): List of b-values.
        dirs_per_shell (list): List of number of directions per shell.

    Returns:
        numpy.ndarray: A gradient table with the specified number of directions per shell.
    """
    tmp_dir = os.path.join(os.getcwd(), 'tmp')
    os.makedirs(tmp_dir, exist_ok=False)
    grad = np.empty((np.sum(dirs_per_shell), 4))
    start = 0

    for bval, dirs in zip(bvals, dirs_per_shell):
        subprocess.run(['dirgen', str(dirs), os.path.join(tmp_dir, 'tmp_dirs.txt'), '-cart'])
        dirs = load_vector(os.path.join(tmp_dir, 'tmp_dirs.txt'))
        subprocess.run(['rm', os.path.join(tmp_dir, 'tmp_dirs.txt')])
        grad[start:start+dirs.shape[0], :-1] = dirs
        grad[start:start+dirs.shape[0], -1] = bval

        start += dirs.shape[0]

    subprocess.run(['rmdir', tmp_dir])

    return grad

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Create a gradient table with a specific number of directions per shell.')
#     parser.add_argument('bvals', type=str, help='Comma-separated list of b-values.')
#     parser.add_argument('dirs_per_shell', type=str, help='Comma-separated list of number of directions per shell.')
#     parser.add_argument('output_file', type=str, help='Output file name.')
#
#     args = parser.parse_args()
#
#     bvals = list(map(int, args.bvals.split(',')))
#     dirs_per_shell = list(map(int, args.dirs_per_shell.split(',')))
#     output_file = args.output_file
#
#     create_gradient_table(bvals, dirs_per_shell, output_file)