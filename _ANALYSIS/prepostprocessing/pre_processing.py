import pandas as pd
import numpy as np
import pickle
from scipy import stats
from sklearn.decomposition import PCA

# will be needed when panels in pandas gets deprecated
# import xarray

# ===================================================================
# TO DO:
#  - Replace soon to be deprecated panels of pandas by xarray package
# ===================================================================


def load_file(path):
    """Load cleaned csv file"""
    file = pd.read_csv(path, index_col=0)
    return file


def normalize(data, total=None):
    """Normalize data to 100%"""
    if total is None:
        total = data.sum(axis=1)
    return data.divide(total, axis=0) * 100


def replace_zero(data, value=0.1):
    """Replace zero values by small value to overcome issues when
    performing log ratio transformation"""
    return data.replace(0.0, value)


def clr(data):
    """Centred log ratio transformation"""

    log_data = np.log(data)
    clr_data = log_data.subtract(np.mean(log_data, axis=1), axis=0)

    assert all(np.isclose(clr_data.sum(axis=1), 0.0))

    return clr_data


def pca(clr_data):
    # Reassign clr_data to overcome copy issues
    data = clr_data
    # Number of components is the minimum of number of rows
    # and number of columns
    pca = PCA()
    pca.fit(data)

    return pca


def pca_variance(pca):
    # Determine PCA expleined variance ratio
    pca_variance = pca.explained_variance_ratio_

    variance_sum = 0
    variance_n_comp = 0

    for v in range(pca.n_components_):
        if variance_sum < 0.95:
            variance_sum += pca.explained_variance_ratio_[v]
            variance_n_comp += 1

    print(
        variance_n_comp,
        "PCA components  out of",
        pca.n_components_,
        "components with variance sum",
        variance_sum,
        "needed for obtaining sum of variance > 0.95",
    )

    return pca_variance


def create_pca_df(pca, clr_data):
    df_pca = pd.DataFrame(
        pca.transform(clr_data),
        columns=["PC" +
                 "{:02}".format(i) for i in range(1, pca.n_components_ + 1)],
        index=clr_data.index,
    )
    return df_pca


def save_obj(obj, name):
    """Save Python instances (e.g. dict) as a binary file to disk
    """
    with open("../_DATA/_obj/" + name + ".pkl", 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_obj_for_python2(obj, name):
    """Save Python instances (e.g. dict) as a binary file to disk
    """
    with open("../_DATA/_obj/" + name + ".pkl", 'wb') as f:
        pickle.dump(obj, f, 1)


def load_obj(name):
    """Save Python instances in binary file format from disk"""
    with open("../_DATA/_obj/" + name + ".pkl", 'rb') as f:
        return pickle.load(f)


def filter_ouliers(df, verbose=False, key=None):
    """Filter outliers based on values with
    absolute z-score bigger than 3"""

    df_copy = df.copy()

    df_filtered = df_copy[(np.abs(stats.zscore(df_copy)) < 3).all(axis=1)]
    if verbose:
        print(key, df_copy.shape, df_filtered.shape)

    return df_filtered
