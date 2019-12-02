import numpy as np
import pandas as pd
import prepostprocessing.pre_processing as preproc


def reverse_clr(reverse_pca, clr, assert_=False):
    """Reverse centered log-ratio transformation"""
    reverse_clr = np.exp(reverse_pca)

    try:
        reverse_clr_df = pd.DataFrame(reverse_clr,
                                      columns=clr.columns,
                                      index=clr.index)
    except ValueError:
        reverse_clr_df = pd.DataFrame(reverse_clr,
                                      columns=clr.columns)

    reverse_clr_df = preproc.normalize(reverse_clr_df)

    if assert_:
        assert all(np.isclose(reverse_clr_df.sum(axis=1), 100.0))

    return reverse_clr_df


def reverse_pca(pca, scores, clr, n_comp=None, assert_=False):
    """Reverse principal compoenents analysis calculation"""
    if n_comp is None:
        n_comp = pca.n_components_

    try:
        reverse_pca = np.dot(scores.iloc[:, :n_comp],
                             pca.components_[:n_comp, :])
    except AttributeError:
        reverse_pca = np.dot(scores[:, :n_comp],
                             pca.components_[:n_comp, :])

    reverse_pca += clr.mean(axis=0).values

    try:
        reverse_pca_df = pd.DataFrame(reverse_pca,
                                      columns=clr.columns,
                                      index=clr.index)
    except ValueError:
        reverse_pca_df = reverse_pca

    if assert_:
        assert all(np.isclose(reverse_pca.sum(axis=1), 0.0))

    return reverse_pca_df


def convert_grid_to_array_of_scores(interpolated, variable=0):

    dfs = {}

    for i in range(1, len(interpolated.keys()) + 1):
        print(i)
        dfs[i] = pd.DataFrame(interpolated[f"PC0{i}"][variable]).copy()

    temp = np.stack((df for df in dfs.values()))
    print(temp.shape)

    scores = np.transpose(temp.reshape(len(interpolated.keys()),
                                       (temp.shape[1] * temp.shape[2])))

    return scores


def convert_points_to_array_of_scores(interpolated, variable=0):

    df = {}

    n_comp = len(interpolated.keys())

    for i in range(1, n_comp + 1):
        df[f"PC0{i}"] = interpolated[f"PC0{i}"][variable].copy()

    temp = np.array((list(df.values())))
    # print(temp.shape)

    scores = np.transpose(temp.reshape(n_comp, (temp.shape[1])))

    return scores
