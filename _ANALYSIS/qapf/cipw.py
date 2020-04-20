import pandas as pd


def convert_to_CIPWFULL_format(df, path, dataset_name="Dataset",
                               index_prefix=None, rock_type="P",
                               rock_suite_column=None, normalization=False,
                               return_resulting_df=False):
    """Converts a pandas DataFrame to a txt file compatible
    as an input file for the CIPWFULL program by AL-Mishwat (2015)

    Parameters:
    -----------
    df : pd.DataFrame
        mineralogy data in the form of oxides
    path : str
        path to where to save resulting file
    dataset_name : str (optional)
        Name opf dataset to use in CIPWFULL
        defaults to 'Dataset'
    index_prefix : str (optional)
        Prefix to use before index
        defaults to None
    rock_type : str (optional)
        Rock type indicator:
            'P' for plutonic
            'V' for volcanic
        defaults to 'P'
    rock_suite_column : str (optional)
        Which column name to use a 'rock suite' in CIPWFULL
        The rock suite acts as a grouping variable
        defaults to None
    normalization : bool (optional)
        Write flag to file whether to let CIPWFULL
        normalize the data before norm calculation
        defaults to False
    return_resulting_df : bool (optional)
        Whether to return the resulting dataframe or not
        defaults to False

    Returns:
    --------
    df_CIPW : pd.DataFrame
        altered mineralogy data according to CIPWFULL input format

    (file is written to disk)

    """

    # Get copy of original df so that it doesn't get altered
    df_CIPW = df.copy()

    # Add prefix to index if required
    if index_prefix:
        df_CIPW.index = f"{index_prefix}" + df_CIPW.index.astype(str)

    # Add additional columns to be moved to index
    df_CIPW["Type"] = "P"
    df_CIPW["Cancrinite"] = 0
    if rock_suite_column:
        df_CIPW["Rock suite"] = df_CIPW[rock_suite_column]
        df_CIPW.drop(rock_suite_column, axis=1)
    else:
        df_CIPW["Rock suite"] = 1

    # Move additional columns to index
    df_CIPW = df_CIPW.set_index(["Type", "Cancrinite", "Rock suite"],
                                append=True)

    # Start writing file (file saving/closing is handled automatically)
    with open(path, 'w') as f:

        # Write a '1' before the dataset name if you want CIPWFULL
        # to normalize the data before the calculation; else write space
        if normalization:
            f.write("1 ")
        else:
            f.write(" ")

        # Write dataset name
        f.write(f"{dataset_name}\n")

        # Write column names and index+data
        # The floating point number need to be formatted so as not
        # to go over the 120 character line limit of CIPWFULL
        df_CIPW.to_csv(f, mode='w', sep=" ", line_terminator="\n",
                       index_label=False, float_format="%2.5f")

        # Write '0' to indicate end of data entries
        f.write("0")

        # Write rock suite names if needed
        if rock_suite_column:
            f.write("\n")
            f.write(rock_suite_column.unique())
            f.write("\n")

        if return_resulting_df:
            return df_CIPW
        else:
            return None


def extract_CIPW_results(path,
                         columns_of_interest=['  QZ', '  OR', '  AB', '  AN'],
                         print_columns=False):
    """Extract the results from CIPWFULL run

    Parameters:
    -----------
    path : str
        Path to results file
    columns_of_interest : list (optional)
        List of minerals to select in results file
        defaults to Q, A, P minerals
    print_columns : bool (optional)
        Print original df's column names
        defaults to False

    Returns:
    --------
    df_final : pd.DataFrame
        Minerals of interest in tabular format
    """

    df = pd.read_csv(path, sep="\t", index_col=0)

    if print_columns:
        print(df.colums)

    # Drop last row which states the column names again
    df = df.iloc[:-1, :]

    # Quary columns in which we're interested
    columns_of_interest = ['  QZ', '  OR', '  AB', '  AN']
    df_query = df.loc[:, columns_of_interest]

    # Convert values to floats
    df_query = df_query.astype(float)

    # Create new dataframe to hold final data
    df_final = pd.DataFrame()

    df_final["Q"] = df_query["  QZ"]
    df_final["P"] = df_query["  AN"] + 0.95 * df_query["  AB"]
    df_final["K"] = df_query["  OR"] + 0.05 * df_query["  AB"]

    return df_final
