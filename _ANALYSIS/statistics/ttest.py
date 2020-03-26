import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, t


def perform_ttest(variogram_results):
    ttest_results = {}

    for key, value in variogram_results.items():
        temp = {}

        for i, item in enumerate(value[3]):
            ttest_stat, p_value = ttest_1samp(item, 0)
            ttest_crit = t.isf(0.05, df=len(item) - 1)
            n_pairs = value[2][i]
            temp[i] = [ttest_stat,
                       ttest_crit,
                       np.abs(ttest_stat) > ttest_crit,
                       p_value / 2,
                       n_pairs,
                       value[0][i]
                       ]

        ttest_results[key] = temp

    return ttest_results


def convert_ttest_results_to_dict_of_df(ttest_results):
    ttest_df_dict = {}

    for key, value in ttest_results.items():

        temp_df = pd.DataFrame.from_dict(value, orient='index')
        temp_df.columns = ["ttest_stat",
                           "ttest_crit",
                           "abs(stat) > crit",
                           "p-value",
                           "n_pairs",
                           "lag"
                           ]
        ttest_df_dict[key] = temp_df

    return ttest_df_dict


def convert_summary_test_results_to_df(ttest_summary,
                                       combinations=["#PCs",
                                                     "search_radius"],
                                       variable_names=["ttest_stat",
                                                       "ttest_crit",
                                                       "abs(stat) > crit"],
                                       order=["#PCs",
                                              "search_radius"],):

    columns = variable_names.copy()
    columns.extend(combinations)

    ttest_summary_df = pd.DataFrame.from_dict(ttest_summary, orient='index') \
                                   .reset_index()
    ttest_summary_df["#PCs"] = \
        list(map(int, ttest_summary_df["index"].str.split("_").str[0]))
    ttest_summary_df["search_radius"] = \
        list(map(int, ttest_summary_df["index"].str.split("_").str[1]))

    ttest_summary_df = ttest_summary_df.drop(["index"], axis=1)
    # print(ttest_summary_df)
    ttest_summary_df = ttest_summary_df.sort_values(order) \
                                       .reset_index(drop=True)
    # print(columns)
    ttest_summary_df.columns = columns
    order.extend(variable_names)
    ttest_summary_df = ttest_summary_df[order]

    return ttest_summary_df


def convert_summary_test_results_to_df_original(
    ttest_summary,
    order=["#PCs"],
    variable_names=["ttest_stat",
                    "ttest_crit",
                    "abs(stat) > crit"]):

    columns = variable_names.copy()
    columns.extend(order)

    ttest_summary_df = pd.DataFrame.from_dict(ttest_summary, orient='index') \
                                   .reset_index()
    ttest_summary_df["#PCs"] = list(map(int, ttest_summary_df["index"]))

    ttest_summary_df = ttest_summary_df.drop(["index"], axis=1)
    ttest_summary_df = ttest_summary_df.sort_values(order) \
                                       .reset_index(drop=True)
    ttest_summary_df.columns = columns
    order.extend(variable_names)
    ttest_summary_df = ttest_summary_df[order]

    return ttest_summary_df


def save_ttest_dict_of_df_to_Excel(ttest_df_dict, pluton, save_abbrev):
    writer = \
        pd.ExcelWriter(f"../_RESULTS/ttests/{pluton}/ttest_{save_abbrev}.xlsx")

    for key, value in ttest_df_dict.items():
        value.to_excel(writer, key)
    writer.save()
    writer.close()
