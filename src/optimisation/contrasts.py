import numpy as np

import numpy as np

def free_water_contrast(ad_range, rd_range):
    """
    Calculate the free water contrast matrix.

    This function generates a matrix representing the contrast for free water based on axial diffusivity (AD) and
    radial diffusivity (RD) ranges. The contrast is set to 1 if the conditions 1000 * AD >= 2.6, 1000 * RD >= 2.6,
    and AD >= RD are met, indicating the presence of free water.

    Parameters:
    - ad_range (list or np.array): A list or array of AD values.
    - rd_range (list or np.array): A list or array of RD values.

    Returns:
    - np.array: A 2D numpy array representing the free water contrast.
    """
    out = np.zeros((len(ad_range), len(rd_range)))

    for i, ad in enumerate(ad_range):
        for j, rd in enumerate(rd_range):
            if 1000 * ad >= 2.6 and 1000 * rd >= 2.6 and ad >= rd:
                out[i, j] = 1

    return out


def intra_axonal_contrast(ad_range, rd_range, with_isotropic=True):
    """
    Calculate the intra-axonal contrast matrix.

    This function generates a matrix representing the contrast for intra-axonal spaces based on radial diffusivity (
    RD) ranges. The contrast is calculated using an exponential decay function applied to the RD values, normalized
    to the maximum value in the resulting matrix.

    Parameters: - ad_range (list or np.array): A list or array of AD values. Not directly used in calculations but
    required for consistent interface. - rd_range (list or np.array): A list or array of RD values.

    Returns:
    - np.array: A 2D numpy array representing the normalized intra-axonal contrast.
    """
    out = np.zeros((len(ad_range), len(rd_range)))
    for i in range(len(rd_range)):
        if with_isotropic:
            out[i, :i + 1] = [np.exp(-5 * j) for j in 1000 * rd_range[:i + 1]]
        else:
            out[i, :i] = [np.exp(-5 * j) for j in 1000 * rd_range[:i]]
    return out / np.max(out)

def extra_axonal_contrast(ad_range, rd_range):
    """
    Calculate the extra-axonal contrast matrix.

    This function generates a matrix representing the contrast for extra-axonal spaces based on axial diffusivity (
    AD) and radial diffusivity (RD) ranges. It is calculated as the complement to the sum of free water and
    intra-axonal contrasts, effectively representing the remaining contrast space.

    Parameters:
    - ad_range (list or np.array): A list or array of AD values.
    - rd_range (list or np.array): A list or array of RD values.

    Returns:
    - np.array: A 2D numpy array representing the extra-axonal contrast.
    """
    return 1 - free_water_contrast(ad_range, rd_range) - intra_axonal_contrast(ad_range, rd_range)

def get_contrast(fs, ad, rd, weighting_function, *args):
    """
    Calculate the weighted contrast based on a given weighting function.

    This function computes the contrast by applying a weighting function to the axial diffusivity (AD) and radial
    diffusivity (RD) values, then multiplying the resulting weights with the signal fractions (fs) and summing over
    the last two dimensions.

    Parameters:
    - fs (np.array): A numpy array representing the signal fractions.
    - ad (np.array): A numpy array of axial diffusivity values.
    - rd (np.array): A numpy array of radial diffusivity values.
    - weighting_function (function): A function that takes AD and RD and returns a weight matrix.

    Returns:
    - float: The sum of the weighted contrast over the specified axes.
    """
    # Calculate weights using the provided weighting function
    weights = weighting_function(ad, rd, *args)

    # Return the sum of the product of weights and signal fractions over the last two dimensions
    return np.sum(weights * fs, axis=(-1, -2))