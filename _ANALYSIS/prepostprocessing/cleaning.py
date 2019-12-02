import numpy as np
import pandas as pd
import re


def match_coordinates(new, old, new_by, old_by):
    """Match new and old coordinates"""

    score = 0

    # Loop over all possibilities
    for n_by in new_by:
        for o_by in old_by:
            for n_asc in [True, False]:
                for o_asc in [True, False]:
                    new_sorted = new.sort_values(by=n_by, ascending=n_asc)
                    old_sorted = old.sort_values(by=o_by, ascending=o_asc)

                    # Calculate correlation coefficient
                    corrcoef = np.corrcoef(new_sorted[n_by],
                                           old_sorted[o_by])[0][1]

                    # Save highest correleation coefficient and according
                    # parameters
                    if np.abs(corrcoef > score):
                        score = corrcoef
                        score_params = {"new_by": n_by,
                                        "old_by": o_by,
                                        "new_ascending": n_asc,
                                        "old_ascending": o_asc}

    return score, score_params


def merge_coordinates(new, old, new_by, old_by, new_ascending, old_ascending):
    """Merge mineralogy data with new coordinates according to matching
    of old to new coordinates"""

    # Sort new and old coordinate data according to 'score_params' of matching
    new_sorted = new.sort_values(by=new_by, ascending=new_ascending)
    old_sorted = old.sort_values(by=old_by, ascending=old_ascending)

    # Merge sorted data from old coordinate DataFrame with new coordinates
    # into new 'merge' DataFrame
    merge = new_sorted.copy()

    for col in old_sorted.columns[2:]:
        merge[col] = old_sorted[col].values

    return merge


def normalize(data, total=None):
    """Normalize data to 100%"""
    if total is None:
        total = data.sum(axis=1)
    return data.divide(total, axis=0) * 100


def save_csvs(combined, coordinates, mineralogy, path):
    """Save combined, coordinates and mineralogy data to seperate csv
    files"""
    combined.to_csv(f"{path}_combined_cleaned.csv",
                    header=True,
                    index=True)

    coordinates.to_csv(f"{path}_coordinates_cleaned.csv",
                       header=True,
                       index=True)

    mineralogy.to_csv(f"{path}_mineralogy_cleaned.csv",
                      header=True,
                      index=True)

    return f"Saved succesfully to {path}..."


def test_cumulative_constraints(mineralogy):
    """Check cumulative data constraints"""

    # constant sum constraint
    assert all(np.isclose(mineralogy.sum(axis=1), 100.0))

    # non-negativity constraint
    assert all(mineralogy >= 0.0)

    return "constant sum and non-negativity constraints verified"


def combine_rest_minerals(df, main=["Q", "P", "K"]):
    """Combine all mineral classes except for the main ones into one
    category"""

    QPK_df = df[main]
    rest_df = df.drop(main, axis=1)

    rest_sum = pd.DataFrame(rest_df.sum(axis=1))
    rest_sum.columns = ["Others"]

    df_new = pd.concat([QPK_df, rest_sum], axis=1)

    return df_new


def dms2dec(dms_str):

    """Return decimal representation of DMS

    Converting Degrees, Minutes, Seconds formatted coordinate strings to decimal.
    Formula:
    DEC = (DEG + (MIN * 1/60) + (SEC * 1/60 * 1/60))
    Assumes S/W are negative.

    Examples:
    ---------
    >>> dms2dec(utf8(48°53'10.18"N))
    48.8866111111F

    >>> dms2dec(utf8(2°20'35.09"E))
    2.34330555556F

    >>> dms2dec(utf8(48°53'10.18"S))
    -48.8866111111F

    >>> dms2dec(utf8(2°20'35.09"W))
    -2.34330555556F

    """

    dms_str = re.sub(r'\s', '', dms_str)

    sign = -1 if re.search('[swSW]', dms_str) else 1

    numbers = list(filter(len, re.split('\D+', dms_str, maxsplit=4)))

    degree = numbers[0]
    minute = numbers[1] if len(numbers) >= 2 else '0'
    second = numbers[2] if len(numbers) >= 3 else '0'
    frac_seconds = numbers[3] if len(numbers) >= 4 else '0'

    # print(numbers)

    second += "." + frac_seconds

    try:
        float(degree)
        float(minute)
        float(second)
    except ValueError:
        print(numbers)
    return sign * (int(degree) + float(minute) / 60 + float(second) / 3600)
