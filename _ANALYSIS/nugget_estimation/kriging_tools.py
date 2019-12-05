import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import chi2
from itertools import combinations

from scipy.spatial import cKDTree


def calculate_pairwise_distances_values(z, drift=0):
    # This is the part of eq(4.96) in Davis(2002) on p. 255 within the sum
    # (xi - xi+h)² / 2
    g = 0.5 * (pdist(z[:, None], metric='euclidean') - drift) ** 2
    return g


def calculate_pairwise_distances_coordinates(xy):
    d = pdist(xy, metric='euclidean')
    return d


def calculate_pairwise_distances_coordinates_square(xy):
    d = squareform(pdist(xy, metric='euclidean'))
    return d


def calculate_and_sort_square_distance_matrix(xy, subset=["X", "Y"]):
    square_matrix = \
        calculate_pairwise_distances_coordinates_square(xy[subset])
    # Drop first entry of sorted array since it represents distance
    # between the point of interest and itself
    square_matrix_sorted = np.sort(square_matrix)[:, 1:]

    return square_matrix_sorted


def find_points_within_search_radius(square_matrix_sorted, search_radius,
                                     return_values=False):

    condition_matrix = square_matrix_sorted < search_radius

    points_within_search_radius = \
        mask_nd(square_matrix_sorted, condition_matrix)

    if return_values:
        return points_within_search_radius
    elif return_values == 'both':
        return (points_within_search_radius,
                np.sum(condition_matrix, axis=1))
    else:
        return np.sum(condition_matrix, axis=1)


def mask_nd(x, m):
    '''
    https://stackoverflow.com/questions/53918392/mask-2d-array-preserving-shape
    Mask a 2D array and preserve the
    dimension on the resulting array
    ----------
    x: np.array
        2D array on which to apply a mask
    m: np.array
        2D boolean mask
    Returns
    -------
    List of arrays. Each array contains the
    elements from the rows in x once masked.
    If no elements in a row are selected the
    corresponding array will be empty
    '''
    take = m.sum(axis=1)
    return np.split(x[m], np.cumsum(take)[:-1])


def calculate_drift(z):
    drift = np.mean(pdist(z[:, None], metric='euclidean'))
    return drift


def determine_n_lags(coordinates):
    return int(np.floor(np.sqrt(coordinates.shape[0])))


def determine_bin_boundaries_equal_bins(d, nlags=4):

    dmax = np.amax(d)
    dmin = np.amin(d)
    dd = (dmax - dmin) / nlags

    bins = [dmin + n * dd for n in range(nlags)]
    bins.append(dmax + 0.001)

    return bins


def determine_bin_boundaries_equal_samples(d, nlags=4,
                                           threshold_absolute=None,
                                           threshold_percentile=95,
                                           plot_bins=False):
    """Determine bin boundaries when every bin should contain the same number
     of samples, except for the last one which holds all samples that are
    above a certain percentile threshold

    Parameters:
    -----------
    d : array
        Distance array
    nlags : int (optional)
        Number of lag bins to produce; defaults to 4
    threshold_percentile : int (optional)
        Value between 0 and 100 which represents the percentile at
        which the threshold will be calculated; defaults to 95
    plot_bins : Bool
        Whether to plot the resulting bins; defaults to False

    Returns:
    --------
    bins : list (float)
        List of bin boundaries
        The last two values represent the bin which contains
        the values above the calculated threshold
    """

    # Determine threshold and select samples above it
    if threshold_absolute is None:
        threshold = np.percentile(d, threshold_percentile)
    else:
        threshold = threshold_absolute
    d_above_threshold = d[d >= threshold]
    d_above_threshold_size = d_above_threshold.size

    # Calculate number of samples that each regular bin should contain
    dd = np.ceil((d.size - d_above_threshold_size) / (nlags - 1))

    # Boundaries of regular bins are added to bins list
    bins = [np.sort(d)[int(n * dd)] for n in range(nlags - 1)]
    # The final bin between the threshold and the maximum does not
    # contain the same number of samples as do the regular bins
    bins.append(threshold)
    bins.append(d.max() + 0.001)

    if plot_bins:
        for bin in bins:
            plt.vlines(bin, 0, 1)
        plt.show()

    return bins


def calculate_lags_and_semivariance(g, d, bins, bin_type='equal_samples',
                                    nlags=4):
    # Loop over the number of lags to consider
    # e.g. of nlags = 4, the loop will first check points that are at 0 times
    # the lag distance, then 1 times the lag distance, until it reaches 4
    # times the lag distance.

    lags = np.zeros(nlags)
    semivariance = np.zeros(nlags)
    n_pairs = np.zeros(nlags)
    semivariance_values = []

    # TO DO
    # ------------------------------------------------------------------------
    # Build in check to see if bins are of equal width or have equal number of
    # samples (except for final bin) --> Based on the outcome, use different
    # mean or median respectively when calculating the lags
    # ------------------------------------------------------------------------

    for n in range(nlags):

        # Check to see if there are any values of d that are in a certain bin
        if d[(d >= bins[n]) & (d < bins[n + 1])].size > 0:

            # Mean value of the pairwise distances of the coordinates that are
            # in a lag bin
            if bin_type == 'equal_bins':
                lags[n] = np.mean(d[(d >= bins[n]) & (d < bins[n + 1])])
            elif bin_type == 'equal_samples':
                lags[n] = np.median(d[(d >= bins[n]) & (d < bins[n + 1])])
            else:
                raise ValueError("'bin_type' should be 'equal_samples' or\
                                 'equal_bins'")

            # Mean value of the pairwise distances of the values values that
            # are in a lag bin
            # This is the part of eq. 4.96 in Davis (2002) taking the sum
            # dividing by n (thus taking the mean)
            semivariance_values.append(g[(d >= bins[n]) & (d < bins[n + 1])])
            semivariance[n] = np.mean(g[(d >= bins[n]) & (d < bins[n + 1])])

            # Number of pairs of points that are within a lag bin
            n_pairs[n] = (d[(d >= bins[n]) & (d < bins[n + 1])]).size

        else:
            lags[n] = np.nan
            semivariance[n] = np.nan

    return lags, semivariance, n_pairs, semivariance_values


def calculate_cross_pairwise_distances_values(z1, z2):
    """http://spatial-analyst.net/ILWIS/htm/ilwisapp/cross_variogram_algorithm.htm"""
    g1 = pdist(z1[:, None], metric=lambda u, v: v-u)
    g2 = pdist(z2[:, None], metric=lambda u, v: v-u)

    g = 0.5 * g1 * g2

    return g


def calculate_cross_pairwise_distances_values_cdist(z1, z2):
    """http://spatial-analyst.net/ILWIS/htm/ilwisapp/cross_variogram_algorithm.htm"""
    g = cdist(z1[:, None], z2[:, None], lambda u, v: u-v)

    return g


def calculate_variogram_results(mineralogy_pca, coordinates,
                                components=["PC01", "PC02", "PC03"],
                                bin_type="equal_samples", n_lags=8,
                                lags_threshold=None):
    variogram_results = {}

    for PCx, PCy in combinations(components, 2):
        # print(PCx, PCy)

        # g is an array containing half the squared euclidean distances
        # between the value (y) data
        g = calculate_cross_pairwise_distances_values(
                mineralogy_pca.loc[:, PCx],
                mineralogy_pca.loc[:, PCy]
                )

        # d is an array containing the euclidean distances of the coordinate
        # data
        d = calculate_pairwise_distances_coordinates(coordinates[["X", "Y"]])
        if bin_type == 'equal_bins':
            bins = determine_bin_boundaries_equal_bins(d, nlags=n_lags)
        elif bin_type == 'equal_samples':
            bins = determine_bin_boundaries_equal_samples(d, nlags=n_lags, threshold_absolute=lags_threshold)
        else:
            raise ValueError("'bin_type' should be 'equal_samples' or\
                             'equal_bins'")

        lags, semivariance, n_pairs, semivariance_values = \
            calculate_lags_and_semivariance(g, d, bins, nlags=n_lags)

        variogram_results[PCx+PCy] = \
            [lags, semivariance, n_pairs, semivariance_values]

    return variogram_results


def calculate_nugget(pca_replicates_center_trans, verbose=False):
    variance = pca_replicates_center_trans.groupby(level=0).var(ddof=1)
    weights = pca_replicates_center_trans.groupby(level=0).count().values[:, 0]
    euclidian_weights = np.array([np.sum(range(x)) for x in weights])
    total_of_euclidian_weights = np.sum(euclidian_weights)
    euclidean_ratios = euclidian_weights/total_of_euclidian_weights

    if verbose:
        print(euclidian_weights, total_of_euclidian_weights)

    nugget = variance.multiply(euclidean_ratios, axis=0).sum(axis=0)
    return nugget


def calculate_theoretical_nugget(k, N=2000, alpha=0.5):
    """
    Parameters:
    -----------
    N : int
        Number of point counts in thin section
    k : int
        Number of components counted
    alpha : float
        Statistical probability level ([0.0-1.0])

    Returns:
    --------
    clr_var
        Minimal clr variance inherently present aka theoretical nugget
    """

    # Integrity checks
    if N <= 0:
        raise ValueError("'N' must be a positive integer bigger than zero")
    if k <= 1:
        raise ValueError("'k' must a positive integer bigger than one")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("'alpha' must be between 0.0 (excl.) and 1.0 (incl.)")

    # Critical Chi² statistic
    chi_crit = chi2.isf(alpha, k-1)

    # 2nd power equation parameters
    A = k**3 - k**2
    B = -2 * k
    C = -1 - (chi_crit/N) + (k/(k-1))

    # Discriminant
    D = np.sqrt(B**2 - 4*A*C)

    # Solutions
    d1 = (-B + D) / (2 * A)
    d2 = (-B - D) / (2 * A)

    # Compositions at vertex (x1) and centre-fase (x2)
    x1 = np.array(((k-1) * d1, (1/(k-1) - d1)))
    x2 = np.array(((k-1) * d2, (1/(k-1) - d2)))

#     print(x1[0] + x1[1] * (k-1))
    assert np.isclose(x1[0] + x1[1] * (k-1), 1.0)
#     print(x2[0] + x2[1] * (k-1))
    assert np.isclose(x2[0] + x2[1] * (k-1), 1.0)

    # Check with Chi² test statistic
    x1_check = k * (x1-(1/k))**2
    x2_check = k * (x2-(1/k))**2

#     print(N * (x1_check[0] + (k-1) * x1_check[1]))
    assert np.isclose(N * (x1_check[0] + (k-1) * x1_check[1]), chi_crit)
#     print(N * (x2_check[0] + (k-1) * x2_check[1]))
    assert np.isclose(N * (x2_check[0] + (k-1) * x2_check[1]), chi_crit)
#     print(chi_crit)

    # ln
    ln_x1 = np.log(x1)
    ln_x2 = np.log(x2)

    # ln mean
    ln_x1_mean = (ln_x1[0] + (k-1) * ln_x1[1])/k
    ln_x2_mean = (ln_x2[0] + (k-1) * ln_x2[1])/k

    # clr
    clr_x1 = ln_x1 - ln_x1_mean
    clr_x2 = ln_x2 - ln_x2_mean

    # clr squared
    clr_x1_squared = clr_x1 ** 2
    clr_x2_squared = clr_x2 ** 2

    # clr variance
    clr_x1_var = clr_x1_squared[0] + (k-1) * clr_x1_squared[1]
    clr_x2_var = clr_x2_squared[0] + (k-1) * clr_x2_squared[1]

    # clr mean variance
    clr_var = np.mean(np.array((clr_x1_var, clr_x2_var)))

    return clr_var
