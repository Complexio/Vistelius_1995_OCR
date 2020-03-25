from __future__ import print_function

import numpy as np
import pandas as pd
from spm1d.stats.hotellings import hotellings, hotellings2


def get_search_radii(residuals_dictionary):
    radii = []

    for key in residuals_dictionary.keys():
        if int(key.split("_")[-1]) not in radii:
            radii.append(int(key.split("_")[-1]))

    return np.sort(radii)


def perform_Hotellings_test(residuals_clr,
                            search_radii,
                            nuggets=["theoretical", "replicates"],
                            weights=None):

    B = np.array([[0.0, 0.0, 0.0, 0.0]])

    test_results = {}
    test_results_details = {}

    for ncomp in range(1, 4):
        for nugget in nuggets:
            for radius in search_radii:
                key = str(ncomp) + "_" + nugget + "_" + str(radius)

                A = residuals_clr[key].values

                if weights is not None:
                    nvalues = weights[radius].values
                else:
                    nvalues=None
                # test_result = hotellings(A, B, weights=nvalues).inference(0.05)
                try:
                    test_result = hotellings(A, B).inference(0.05)
                except np.linalg.LinAlgError:
                    print(f"LinAlgError on run {ncomp}_{nugget}_{radius}")
                test_results[key] = [test_result.h0reject,
                                     test_result.p,
                                     np.abs(test_result.z) / test_result.zstar,
                                     test_result.alpha]
                test_results_details[key] = test_result

    return test_results, test_results_details


def convert_test_results_to_df(hotelling_results):
    hotelling_results_df = \
        pd.DataFrame.from_dict(hotelling_results, orient='index').reset_index()

    # Set 'index' columns
    hotelling_results_df["nugget_type"] = \
        hotelling_results_df["index"].str.split("_").str[1]
    hotelling_results_df["search_radius"] = \
        list(map(int, hotelling_results_df["index"].str.split("_").str[2]))
    hotelling_results_df["#PCs"] = \
        list(map(int, hotelling_results_df["index"].str.split("_").str[0]))

    hotelling_results_df["H0_reject"] = hotelling_results_df[0]
    hotelling_results_df["p-value"] = hotelling_results_df[1]
    hotelling_results_df["test_stat/test_crit"] = hotelling_results_df[2]
    hotelling_results_df["alpha"] = hotelling_results_df[3]

    hotelling_results_df = \
        hotelling_results_df.drop(["index", 0, 1, 2, 3], axis=1)
    hotelling_results_df = hotelling_results_df.sort_values(
        ["nugget_type", "search_radius", "#PCs"]).reset_index(drop=True)

    return hotelling_results_df


def perform_Hotellings_test_simulations(samples_simulation,
                                        mean_clr,
                                        alpha=0.05):

    B = mean_clr.reshape(1, 4)

    test_results = {}
    test_results_details = {}

    for sample_size, samples in samples_simulation.items():
        if sample_size in [1]:
            continue
        temp = []
        temp_details = []
        for i, item in enumerate(samples):

            A = item.values
            try:
                test_result = hotellings(A, B).inference(alpha)
            except(np.linalg.LinAlgError):
                print(sample_size, i)
                continue
            temp.append(test_result.h0reject)
            temp.append(test_result.p)
            temp.append(test_result.z / test_result.zstar)
            temp.append(alpha)
            temp_details.append(test_result)
        test_results[sample_size] = temp
        test_results_details[sample_size] = temp_details

    return test_results, test_results_details


def perform_empirical_test_simulations(samples_simulation,
                                       mean_clr,
                                       alpha=0.05,
                                       nugget=0.0005):

    B = mean_clr.reshape(1, 4)

    test_results = {}

    for sample_size, samples in samples_simulation.items():
        if sample_size in [1]:
            continue
        temp = []
        for i, item in enumerate(samples):

            A = item.values

            temp.append(np.sum((A - B) ** 2) > nugget)

        test_results[sample_size] = temp

    return test_results
