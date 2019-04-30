import pandas as pd
import numpy as np


##################################################################
###################  Utility/Helper Functions  ###################
##################################################################


def get_propensity_scores(model, data, verbose = False):
    '''
    Utilizes a logistic regression framework to calculate propensity scores
    based on a specified model.

    Parameters
    ----------
    model : string
        a model specification in the form Y ~ X1 + X2 + ... + Xn
    data : Pandas DataFrame
        the data used to calculate propensity scores
    verbose : boolean
        verbosity of the model output

    Returns
    -------
    An array of propensity scores.
    '''
    import statsmodels.api as sm
    glm_binom = sm.formula.glm(formula = model, data = data, family = sm.families.Binomial())
    result = glm_binom.fit()
    if verbose:
        print(result.summary)
    return result.fittedvalues


def get_matched_data(match_ids, raw_data):
    '''
    Subsets the raw data to include data only from the treated cases and
    their respective matched control(s).

    Parameters
    ----------
    match_ids : Pandas DataFrame
        a dataframe of treated case IDs and matched control(s)
    raw_data: Pandas DataFrame
        a dataframe of all of the raw data

    Returns
    -------
    A dataframe containing data only from treated cases and matched control(s).
    '''
    match_ids = flatten_match_ids(match_ids)
    matched_data = raw_data[raw_data.index.isin(match_ids)]
    return matched_data


def make_crosstable(df, var):
    '''
    Makes a frequency-count crosstable for use in chi-square testing.

    Parameters
    ----------
    df : Pandas DataFrame
        a dataframe containing data to be analyzed
    var : string
        the variable to be analyzed

    Returns
    -------
    A Pandas Crosstable object.
    '''
    crosstable = pd.crosstab(df["CASE"], df[var])
    return crosstable


def calc_chi2_2x2(crosstable):
    '''
    Calculates the chi-square statistic, df, and p-value for a 2x2 table.

    Parameters
    ----------
    crosstable : Pandas CrossTab
        the object returned by the make_crosstable() function

    Returns
    -------
    An array containing the resulting chi-square statistic, df, and p-value.
    '''
    from scipy.stats import chi2_contingency
    f_obs = np.array([crosstable.iloc[0][0:2].values,
                      crosstable.iloc[1][0:2].values])
    result = chi2_contingency(f_obs)[0:3]
    round_result = (round(i,4) for i in result)
    return list(round_result)


def calc_chi2_2xC(crosstable):
    '''
    Calculates the chi-square statistic, df, and p-value for a 2xC table.

    Parameters
    ----------
    crosstable : Pandas CrossTab
        the object returned by the make_crosstable() function

    Returns
    -------
    An array containing the resulting chi-square statistic, df, and p-value.
    '''
    from scipy.stats import chi2_contingency
    C = crosstable.shape[1]
    f_obs = np.array([crosstable.iloc[0][0:C].values,
                      crosstable.iloc[1][0:C].values])
    result = chi2_contingency(f_obs)[0:3]
    round_result = (round(i,4) for i in result)
    return list(round_result)


def flatten_match_ids(df):
    '''
    Converts a Pandas DataFrame of matched IDs into a list of those IDs.

    Parameters
    ----------
    df : Pandas Dataframe
        a dataframe consisting of 1 column of treated/case IDs and n columns
        of respective control(s) matched

    Returns
    -------
    A list of treated case and matched control IDs.
    '''
    master_list = []
    master_list.append(df[df.columns[0]].tolist())
    for i in range(1, df.shape[1]):
        master_list.append(df[df.columns[i]].tolist())
    master_list = [item for sublist in master_list for item in sublist]
    return master_list


def write_data(file, df):
    '''
    Writes matched data to file.

    Parameters
    ----------
    file : string
        a file path used to derive the saved file path
    df : Pandas Dataframe
        the dataframe to be written to file.
    '''
    print("\nWriting data to file ...", end = " ")
    save_file = file.split(".")[0] + "_matched_ps.csv"
    df.to_csv(save_file, index = False)
    print("DONE!")
    print()
