import numpy as np
import pandas as pd
from scipy import stats
import pickle


def ecdf(data):
    """Compute Empirical Cumulative Distribution Function (ECDF) for a
       one-dimensional array of measurements."""

    n = len(data)
    data_sorted = np.sort(data)
    ecdf = np.arange(1, n+1) / float(n)

    return data_sorted, ecdf


def clr(data):
    """Centred log ratio transformation"""

    log_data = np.log(data)
    clr_data = log_data.subtract(np.mean(log_data, axis=1), axis=0)

    return clr_data


def alr(data):
    """Additive log ratio transformation"""

    alr_data = np.log(data.divide(1 - data, axis=0))

    return alr_data


def sigmoid_fit(x, x0, k):
    """Sigmoid fit function"""
    y = 1 / (1 + np.exp(-k*(x-x0)))

    return y


def lognormal_fit(x, a, b):
    """Lognormal fit function"""
    y = a + b * np.log(x)

    return y


def exponential_fit(x, a, b):
    """Exponential fit function"""
    y = a * np.exp(b * x)

    return y


def power_law_fit(x, a, b, c):
    """Power law fit function"""
    y = a + b * x**c

    return y


def power_law_fit_fixed(x, a, b):

    y = a + b * 1/x

    return y


def save_obj(obj, name):
    """Save Python instances (e.g. dict) as a binary file to disk
    """
    with open("../_DATA/obj/" + name + ".pkl", 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """Save Python instances in binary file format from disk"""
    with open("../_DATA/obj/" + name + ".pkl", 'rb') as f:
        return pickle.load(f)


def geometrics(data):
    """Calculate geometric mean and geometric standard deviation

    Parameters:
    -----------
    data (array)
        Values to average

    Returns:
    --------
    geo_mean (float)
        Geometric mean
    geo_std (float)
        Geometric standard deviation
    """

    geo_mean = stats.mstats.gmean(data)

    geo_std = np.exp(np.sqrt(np.sum((np.log(data/geo_mean))**2)/len(data)))

    return geo_mean, geo_std


def binning(col, cut_points, labels=None):
    """Binning of values inside DataFrame column based on predetermined
    cut points.

    Parameters:
    -----------
    col : pd.Series
        Column of DataFrame to be binned
    cut_points : list
        Points to use as bin cut offs as int
    labels : list (optional)
        List of label names as str

    Returns:
    --------
    colBin : pd.Series
        Binned DataFrame column
    """

    # Define min and max values:
    minval = col.min()
    maxval = col.max()

    # create list by adding min and max to cut_points
    break_points = [minval] + cut_points + [maxval]

    # if no labels provided, use default labels 0 ... (n-1)
    if not labels:
        labels = range(len(cut_points) + 1)

    # Binning using cut function of pandas
    colBin = pd.cut(col, bins=break_points, labels=labels, include_lowest=True)

    return colBin

# ------------------------------- #
# JUPYTER NOTEBOOKS USEFULL LINES #
# ------------------------------- #

# Load jupyter extension to reload packages before executing user code.
# https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html
# %load_ext autoreload

# Reload all packages (except those excluded by %aimport) every time
# before executing the Python code typed.
# %autoreload 2

# Load extension to profile functions inline
# %load_ext line_profiler

# Example
# %lprun -f simulate_3D_to_1D simulate_3D_to_1D(simulation_size=1000,
# circles_size=3000, lines_size=500)
