import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from pykrige.rk import Krige
from sklearn.metrics import make_scorer, mean_squared_error

# ys = []


# def MSE(y_true, y_pred):
#     global ys
#     ys.append(y_pred)
#     mse = mean_squared_error(y_true, y_pred)
#     return mse


# def return_ys():
#     return ys


def perform_loocv(mineralogy_pca,
                  coordinates_utm,
                  cv_param_dict,
                  n_jobs=-1,
                  verbose_level=5):
    """Perform Leave-One-Out Cross Validation"""

    estimators = {}

    # Do not use the last principal component
    for component in mineralogy_pca.columns.tolist()[:-1]:
        # print(component)

        scorer = make_scorer(mean_squared_error)
        # scorer = make_scorer(mean_squared_deviation_ratio)

        estimator = GridSearchCV(Krige(),
                                 cv_param_dict[component],
                                 verbose=verbose_level,
                                 cv=cv_param_dict[component]["n_closest_points"][0],
                                 iid=False,
                                 n_jobs=n_jobs,
                                 return_train_score=True,
                                 scoring=scorer)

        estimator.fit(X=np.stack((coordinates_utm["X"],
                                  coordinates_utm["Y"]), axis=1),
                      y=mineralogy_pca[component])

        print(estimator)

        estimators[component] = estimator

    return estimators


def perform_loocv_clr(mineralogy_clr,
                      coordinates_utm,
                      variogram_model_parameters,
                      number_of_control_points,
                      n_lags=12,
                      search_radii=[500, 1000],
                      verbose_level=5):
    """Perform Leave-One-Out Cross Validation"""

    estimators = {}

    for component, clr_solution in mineralogy_clr.items():

        param_dict = {"method": ["ordinary"],
                      "variogram_model": ["exponential"],
                      "variogram_model_parameters":
                      [variogram_model_parameters[component]],
                      "nlags": [n_lags],
                      "weight": [True],
                      "n_closest_points": [number_of_control_points],
                      "search_radius": search_radii
                      }

        scorer = make_scorer(mean_squared_error)
        # scorer = make_scorer(mean_squared_deviation_ratio)

        estimator = GridSearchCV(Krige(),
                                 param_dict,
                                 verbose=verbose_level,
                                 cv=number_of_control_points,
                                 iid=False,
                                 n_jobs=-1,
                                 return_train_score=True,
                                 scoring=scorer)

        estimator.fit(X=np.stack((coordinates_utm["X"],
                                  coordinates_utm["Y"]), axis=1),
                      y=mineralogy_clr[component])

        estimators[component] = estimator

    return estimators


def convert_scores_to_df(estimators):
    """Convert cross validation scoring output to a pandas DataFrame"""
    CV_results = {}

    for component, value in estimators.items():
        CV_results_component = \
            pd.DataFrame(value.cv_results_)[['rank_test_score',
                                             'mean_test_score',
                                             'std_test_score',
                                             'mean_train_score',
                                             'param_method',
                                             'param_variogram_model',
                                             'param_search_radius']
                                            ].sort_values("rank_test_score",
                                                          ascending=False)

        CV_results[component] = CV_results_component

    return CV_results


def print_best_scores(CV_results):
    for component, value in CV_results.items():
        print(component, value.iloc[0], "\n", sep="\n")
