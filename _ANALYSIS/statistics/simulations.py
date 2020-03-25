import numpy as np
import pandas as pd
from numpy.random import RandomState

from prepostprocessing import post_processing as postproc


def select_random_sample_locations(grid, mask, seed, max_sample_size):

    randomizer = RandomState(seed)

    grid_spacing = grid[0][1] - grid[0][0]

    n = 0
    counter = 0
    random_samples = []

    while len(random_samples) < max_sample_size:
        X_randint = randomizer.randint(0, grid[0].shape[0] - 1, 1)[0]
        Y_randint = randomizer.randint(0, grid[1].shape[0] - 1, 1)[0]
    #     print(X_randint, Y_randint)
        counter += 1

        if not mask[Y_randint, X_randint]:
            random_samples.append([X_randint, Y_randint])
            n += 1
        # Fail safe in case while loop gets out of control
        if counter > 10_000_000:
            break

        if n == 10:
            print(seed, counter, n)

    random_samples_array = np.array(random_samples)

    X_rand_indices = random_samples_array[:, 0]
    Y_rand_indices = random_samples_array[:, 1]

    x = 1 if randomizer.random_sample() < 0.5 else -1
    X_random_shifted = \
        grid[0][X_rand_indices] + \
        grid_spacing / 2 * randomizer.random_sample() * x

    y = 1 if randomizer.random_sample() < 0.5 else -1
    Y_random_shifted = \
        grid[1][Y_rand_indices] + \
        grid_spacing / 2 * randomizer.random_sample() * y

    return X_random_shifted, Y_random_shifted


def create_random_samples(variograms, critical_distance, n, max_points, n_comp,
                          grid, mask, pca, clr, CV_errors=None):
    print(max_points)
    grid_spacing = grid[0][1] - grid[0][0]

    # Initialize arrays for random sample coordinates
    random_locations = []
    random_points = []

    # Perform random location sampling, kriging and inverse pca
    # for i in range(n_runs):
        # print(i)

    # Step 1 - Select random sample locations
    X_random_shifted, Y_random_shifted = \
        select_random_sample_locations(grid, mask, n, max_points)
    random_locations.append([X_random_shifted, Y_random_shifted])

    # Step 2a - Krige for set number of components
    kriged_points_pca = []

    for ncomp in range(1, n_comp + 1):
        kriged_points_pca.append(
            variograms[f"PC0{ncomp}"].execute(
                "points",
                X_random_shifted,
                Y_random_shifted,
                backend="loop",
                n_closest_points=clr.shape[0],
                search_radius=critical_distance + grid_spacing/2)
            )

    # Step 2b - Convert to PC scores
    kriged_points_pca_dict = {}

    for j, item in enumerate(kriged_points_pca, start=1):
        kriged_points_pca_dict[f"PC0{j}"] = item
    if n == 0:
        print(kriged_points_pca_dict.keys())

    kriged_points_pca_dict_ = kriged_points_pca_dict

    # Step 3
    # For each random sample location take a random sample from a
    # normal distribution with the kriged PC score as the mean and
    # the square root of the CV error for that PC as the standard
    # deviation
    if CV_errors is not None:
        randomizer = RandomState(n)

        kriged_points_pca_dict_normal_sample = {}

        for key, value in kriged_points_pca_dict.items():
            value_normal_sample = \
                randomizer.normal(loc=value[0], scale=np.sqrt(CV_errors[key]))
                # randomizer.normal(loc=value[0], scale=CV_errors[key])
            # if i == 0:
            #     print(value[0], value_normal_sample)

            kriged_points_pca_dict_normal_sample[key] = \
                pd.DataFrame(value_normal_sample)

        kriged_points_pca_dict_ = kriged_points_pca_dict_normal_sample

    # if i == 0:
    #     print(kriged_points_pca_dict_.keys())
    #     print(kriged_points_pca_dict_["PC01"].shape)

    # Step 4 - Inverse pca based on set number of components
    kriged_pca_scores = \
        postproc.convert_points_to_array_of_scores(kriged_points_pca_dict_,
                                                   variable=0)

    kriged_clr = postproc.reverse_pca(pca,
                                      kriged_pca_scores,
                                      clr,
                                      n_comp=n_comp
                                      )

    if isinstance(kriged_clr, np.ndarray):
        random_points.append(kriged_clr)
    else:
        kriged_clr = kriged_clr.reset_index(drop=True)
        random_points.append(kriged_clr.values)

    return np.array(random_points), random_locations
        # [value[0], CV_errors[key], value_normal_sample]
