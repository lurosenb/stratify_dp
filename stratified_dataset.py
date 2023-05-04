import numpy as np
import pandas as pd

class StratifiedDataset:
    
    def __init__(self, df, strata_cols, categorical_columns=[], default_bins=10, smallest_strata=0.001):
        # create hash mapping for categorical columns
        self.num_to_cat = {}
        
        # convert dataframe to numeric
        self.df = self.force_data_categorical_to_numeric(df.copy(), cat_columns=categorical_columns)
        # add a column to the dataframe to store the strata index
        self.df['strata_index'] = -1
        self.strata_cols = strata_cols
        self.default_bins = default_bins
        self.smallest_strata = smallest_strata

        # create strata hash mapping 
        self.strata_to_id = {}
        self.id_to_strata = {}

        # for each row in df, add its strata to the hash if its not present
        strata_index = 0
        for index, row in df.iterrows():
            stratum = self.calculate_strata_for_row(row)
            if stratum not in self.strata_to_id:
                self.strata_to_id[stratum] = strata_index
                # reverse mapping
                self.id_to_strata[strata_index] = stratum
                strata_index += 1
            # tag the row with its strata index
            self.df.loc[index, 'strata_index'] = self.strata_to_id[stratum]

    def calculate_strata_for_row(self, row):
        # map a row to a stratum
        stratum = []
        for col in self.strata_cols:
            stratum.append(row[col])
        return tuple(stratum)
    
    def get_strata_count(self):
        return len(self.strata_to_id)
    
    def strata_size_filter(self, strata, verbose=False):
        # check if a stratum is too small
        check = strata.shape[0] > self.smallest_strata * self.df.shape[0]
        if verbose:
            print('Strata size:', strata.shape[0], 'Smallest strata size:', self.smallest_strata * self.df.shape[0])
        return check

    def get_strata_dfs(self, limit_size=False, remove_strata_index=True):
        # return a list of dataframes, one for each stratum
        strata_dfs = []
        for strata_index in range(self.get_strata_count()):
            strata = self.df[self.df['strata_index'] == strata_index]
            if limit_size:
                if self.strata_size_filter(strata):
                    strata_dfs.append(strata)
            else:
                strata_dfs.append(strata)
        if remove_strata_index:
            for strata_df in strata_dfs:
                strata_df.drop('strata_index', axis=1, inplace=True)
        return strata_dfs
    
    def force_data_categorical_to_numeric(self, df, cat_columns=[]):
        # convert columns to categorical if they are not already
        for col in cat_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
                # save mapping back to original values
                self.num_to_cat[col] = dict(enumerate(df[col].cat.categories))
                df[col] = df[col].cat.codes
        return df