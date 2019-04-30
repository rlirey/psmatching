import pandas as pd
import numpy as np
from psmatching.utilities import *


####################################################
###################  Base Class  ###################
####################################################


class PSMatch(object):
    '''
    Parameters
    ----------
    file : string
        The file path of the data; assumed to be in .csv format.
    model : string
        The model specification for calculating propensity scores; in the format Y ~ X1 + X2 + ... + Xn
    k : string
        The number of controls to be matched to each treated case.
    '''

    def __init__(self, file, model, k):
        self.file = file
        self.model = model
        self.k = int(k)


    def prepare_data(self, **kwargs):
        '''
        Prepares the data for matching.

        Parameters
        ----------
        file : string
            The file path of the data to be analyzed. Assumed to be in .csv format.

        Returns
        -------
        A Pandas DataFrame containing raw data plus a column of propensity scores.
        '''
        # Read in the data specified by the file path
        df = pd.read_csv(self.file)
        df = df.set_index("OPTUM_LAB_ID")
        # Obtain propensity scores and add them to the data
        print("\nGetting propensity scores ...", end = " ")
        propensity_scores = get_propensity_scores(model = self.model, data = df, verbose = False)
        print("DONE!")
        df["PROPENSITY"] = propensity_scores
        # Assign the df attribute to the Match object
        self.df = df


    def match(self, **kwargs):
        '''
        Performs propensity score matching.

        Parameters
        ----------
        df : Pandas DataFrame
            the attribute returned by the prepare_data() function

        Returns
        -------
        matches : Pandas DataFrame
            the Match object attribute describing which control IDs are matched
            to a particular treatment case.
        matched_data: Pandas DataFrame
            the Match object attribute containing the raw data for only treatment
            cases and their matched controls.
        '''
        # Assert that the Match object has a df attribute
        if not hasattr(self, 'df'):
            raise AttributeError("%s does not have a 'df' attribute." % (self))

        # Assign treatment group membership
        groups = self.df.CASE
        propensity = self.df.PROPENSITY
        groups = groups == groups.unique()[1]
        n = len(groups)
        n1 = groups[groups==1].sum()
        n2 = n-n1
        g1, g2 = propensity[groups==1], propensity[groups==0]

        if n1 > n2:
            n1, n2, g1, g2 = n2, n1, g2, g1

        # Randomly permute the treatment case IDs
        m_order = list(np.random.permutation(groups[groups==1].index))
        matches = {}
        k = int(self.k)

        # Match treatment cases to controls based on propensity score differences
        print("\nRunning match algorithm ... ", end = " ")
        for m in m_order:
            # Calculate all propensity score differences
            dist = abs(g1[m]-g2)
            array = np.array(dist)
            # Choose the k smallest differences
            k_smallest = np.partition(array, k)[:k].tolist()
            keep = np.array(dist[dist.isin(k_smallest)].index)

            # Break ties via random choice, if ties are present
            if len(keep):
                matches[m] = list(np.random.choice(keep, k, replace=False))
            else:
                matches[m] = [dist.idxmin()]

            # Matches are made without replacement
            g2 = g2.drop(matches[m])

        # Prettify the results by consolidating into a DataFrame
        matches = pd.DataFrame.from_dict(matches, orient="index")
        matches = matches.reset_index()
        column_names = {}
        column_names["index"] = "CASE_ID"
        for i in range(k):
            column_names[i] = str("CONTROL_MATCH_" + str(i+1))
        matches = matches.rename(columns = column_names)

        # Extract data only for treated cases and matched controls
        matched_data = get_matched_data(matches, self.df)
        print("DONE!")
        write_data(self.file, self.df)

        # Assign the matches and matched_data attributes to the Match object
        self.matches = matches
        self.matched_data = matched_data


    def evaluate(self, **kwargs):
        '''
        Conducts chi-square tests to verify statistically that the cases/controls
        are well-matched on the variables of interest.
        '''
        # Assert that the Match object has 'matches' and 'matched_data' attributes
        if not hasattr(self, 'matches'):
            raise AttributeError("%s does not have a 'matches' attribute." % (self))
        if not hasattr(self, 'matched_data'):
            raise AttributeError("%s does not have a 'matched_data' attribute." % (self))

        # Get variables of interest for analysis
        variables = self.df.columns.tolist()[0:-2]
        results = {}
        print()

        # Evaluate case/control match for each variable of interest
        for var in variables:
            crosstable = make_crosstable(self.df, var)
            if len(self.df[var].unique().tolist()) <= 2:
                p_val = calc_chi2_2x2(crosstable)[1]
            else:
                p_val = calc_chi2_2xC(crosstable)[1]
            results[var] = p_val
            print("\t\t" + var, end = "")
            if p_val < 0.05:
                print(": FAILED!")
            else:
                print(": PASSED!")

        if True in [i < 0.05 for i in results.values()]:
            print("\n\t\tAt least one variable failed to match!")
        else:
            print("\n\t\tAll variables were successfully matched!")
































































