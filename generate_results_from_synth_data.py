#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from stratified_dataset import ParallelStratifiedSynthesizer
from snsynth.mst import MSTSynthesizer
from snsynth.aim import AIMSynthesizer
from gem_synthesizer import GEMSynthesizer
import dill
from helpers.data_utils import get_employment, calculate_dimensionality
import itertools
import os
from IPython.display import clear_output
from stratified_dataset import StratifiedDataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import false_positive_rate, false_negative_rate, equalized_odds_ratio, demographic_parity_ratio
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from helpers.send_emails import send_email

all_data, features, target, group = get_employment()

def load_pickled_model(filename, torch=False):
    with open(filename, "rb") as file:
        model = dill.load(file)
    return model


# In[2]:


df = all_data.copy()

data_dimensionality = calculate_dimensionality(df)
print("Dimensionality of the data before:", data_dimensionality)

df = df.drop(columns=['CIT', 'MIG', 'DEAR', 'DEYE', 'NATIVITY', 'ANC'])

data_dimensionality = calculate_dimensionality(df)
print("Dimensionality of the data after:", data_dimensionality)


# In[3]:


def evaluate_on_dataframes(train_df, test_df, target_col = 'ESR'):
    # Feature columns
    feature_cols = [col for col in train_df.columns if col != target_col]

    # Convert all columns to categorical
    for col in train_df.columns:
        train_df[col] = train_df[col].astype('category')

    # Prepare the dataset
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Train the classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy
    
def min_max_eval(train_df, test_df, strata_cols, target_col = 'ESR'):
    # Feature columns
    feature_cols = [col for col in train_df.columns if col != target_col]

    # Convert all columns to categorical
    for col in train_df.columns:
        train_df[col] = train_df[col].astype('category')

    # Prepare the dataset
    combinations = []
    for i in range(1, len(strata_cols) + 1):
        combinations.extend(list(itertools.combinations(strata_cols, i)))
    
    accuracies = []
    for combination in combinations:
        keys_strat = synth_df_strat[list(combination)].value_counts().keys()
        for key in keys_strat:
            for var in keys_strat.names:
                if list(keys_strat.names) == ['SEX', 'RAC1P']:
                    subset = synth_df_strat.loc[(synth_df_strat['SEX'] == key[0]) & (synth_df_strat['RAC1P'] == key[1])]
                elif list(keys_strat.names) == ['RAC1P']:
                    subset = synth_df_strat.loc[(synth_df_strat['RAC1P'] == key[0])]
                else:
                    subset = synth_df_strat.loc[(synth_df_strat['SEX'] == key[0])]
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = subset[feature_cols]
            y_test = subset[target_col]

            # Train the classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = clf.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append((accuracy, (str(list(combination)), key)))
            print((accuracy, (str(list(combination)), key)))
    min_tuple = min(accuracies, key=lambda x: x[0])
    max_tuple = max(accuracies, key=lambda x: x[0])

    return (min_tuple[0], max_tuple[0])

from sklearn.metrics import accuracy_score
from fairlearn.metrics import false_positive_rate, false_negative_rate, equalized_odds_ratio, demographic_parity_ratio
from fairlearn.metrics import MetricFrame

def evaluate_on_dataframes_with_fairlearn(train_df, test_df, target_col = 'ESR'):
    # Feature columns
    feature_cols = [col for col in train_df.columns if col != target_col]

    # # Convert all columns to categorical
    for col in train_df.columns:
        train_df[col] = train_df[col].astype('float')

    for col in test_df.columns:
        test_df[col] = test_df[col].astype('float')

    test_df = test_df.dropna(subset=['SEX', 'RAC1P'])
    # Prepare the dataset
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Train the classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Compute fairness metrics
    metrics = MetricFrame({
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
    }, y_test.values, y_pred, sensitive_features=test_df[['SEX','RAC1P']])

    # Compute difference and ratio
    m_dif = metrics.difference()
    m_ratio = metrics.ratio()
    fpr_difference = m_dif['false_positive_rate']
    fnr_difference = m_dif['false_negative_rate']

    # Define sensitive features
    sensitive_features_test = X_test[['SEX','RAC1P']]  # Replace 'strata_cols' with the actual column name(s)

    # Compute equalized odds ratio
    dpr = demographic_parity_ratio(y_true=y_test, 
                            y_pred=y_pred, 
                            sensitive_features=sensitive_features_test)

    results = {
        'Accuracy': accuracy,
        'False_positive_rate': metrics.overall['false_positive_rate'],
        'False_negative_rate': metrics.overall['false_negative_rate'],
        'FPR_difference': fpr_difference,
        'FNR_difference': fnr_difference,
        'Demographic_parity_ratio': dpr,
    }
    
    return results


# In[4]:

def get_subgroup_key(group, groupby_cols):
    key = []
    for col in groupby_cols:
        unique_values = group[col].unique()
        if len(unique_values) == 1:
            key.append((col, unique_values[0]))
        else:
            print(f"More than one unique value found for column '{col}' in the given group.")
            print(f"Unique values found: {unique_values}")
            raise ValueError(f"More than one unique value found for column '{col}' in the given group.")
    return tuple(key)

def create_subgroups_dict(X, groupby_cols):
    subgroups = {}
    for _, group in X.groupby(groupby_cols):
        if not group.empty:
            key = get_subgroup_key(group, groupby_cols)
            subgroups[key] = group
        else:
            print('This weird thing happens sometimes where a group is empty. Not sure why.')
    return subgroups
    
def parity_error_synth_data(X, X_prime, groupby_cols, f, omega):
    subgroups_real = create_subgroups_dict(X, groupby_cols)
    subgroups_synth = create_subgroups_dict(X_prime, groupby_cols)
    f_values_real = []
    f_values_synth = []

    # Calculate f and M values for each stratum
    # for key in keys_for_comparison:
    for key, s in subgroups_real.items():
        s = subgroups_real[key]
        f_value_real = f(s)
        f_values_real.append(f_value_real)
        
        if key in subgroups_synth:
            f_value_synth = f(subgroups_synth[key])
        else:
            print(subgroups_real.keys())
            print(subgroups_synth.keys())
            print((f'Should not happen: {key} not in subgroups_synth'))
            f_value_synth = f(X_prime)

        f_values_synth.append(f_value_synth)

    # Calculate the global f and M values
    f_global = f(X)
    f_synth_global = f(X_prime)

    # Compute the parity error
    beta = omega * (abs(f_global - f_synth_global) / f_global) + sum([(abs(t - s) / t) for t, s in zip(f_values_real, f_values_synth)])

    return beta

def calculate_disparity(real_train_df, synth_df, strata_cols, func):
    assert len(strata_cols) > 0, "strata_cols must be a list with at least one element"

    # Create multi-index DataFrames grouped by strata_cols
    real_grouped = real_train_df.groupby(strata_cols)
    synth_grouped = synth_df.groupby(strata_cols)

    # Initialize disparity as negative infinity
    max_disparity = float('-inf')
    max_key = None

    # Iterate over unique combinations of strata_cols
    for key in real_grouped.groups.keys():
        # Check if the group also exists in the synthetic data
        if key in synth_grouped.groups.keys():
            real_group = real_grouped.get_group(key)
            synth_group = synth_grouped.get_group(key)

            # Apply the function to the real and synthetic groups
            real_result = func(real_group)
            synth_result = func(synth_group)

            # Calculate the absolute difference normalized
            disparity = abs((real_result - synth_result) / real_result) 
            # If this disparity is greater than the current maximum, update maximum
            for disp in disparity:
                if not np.isinf(disp):
                    if disp > max_disparity:
                        max_disparity = disp
                        max_key = key
                        
    return max_disparity, max_key


# In[5]:


def mean_f(df):
    return df.astype(float).mean().values

def add_row_to_performance_df(performance_df, synth_class, synth_df, epsilon, real_train_df, real_test_df, name_combo, omega = 0.2, strata_cols = ['SEX', 'RAC1P']):
    result_dict = evaluate_on_dataframes_with_fairlearn(synth_df, real_test_df)
    row_dict = {
        'Synthesizer': synth_class.__name__ + "_" + name_combo,
        'Epsilon': epsilon,
    }
    
    for k, v in result_dict.items():
        row_dict[k] = v

    for combination in combinations:
        strata_cols = list(combination)

        # Calculate parity error
        parity_error = parity_error_synth_data(real_train_df, synth_df, strata_cols, mean_f, omega)
        row_dict[str(strata_cols)+"_parity_error"] = parity_error

        # Calculate max disparity
        max_disparity = calculate_disparity(real_train_df, synth_df, strata_cols, mean_f)
        row_dict[str(strata_cols)+"_max_disparity"] = max_disparity


    performance_df = performance_df.append(row_dict, ignore_index=True)
    return performance_df


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')
seed = 1

synthesizers = [AIMSynthesizer]
epsilons = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
omega = 1/12 # (1/k is the default value in the paper)
strata_cols = ['SEX', 'RAC1P']

# Generate all possible combinations of the given column names
combinations = []
for i in range(1, len(strata_cols) + 1):
    combinations.extend(list(itertools.combinations(strata_cols, i)))

def step_email(iter):
    subject = f"COMPLETE: Iter {iter})"
    body = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_email_target = 'lr2872@nyu.edu'
    send_email(subject, body, log_email_target)

def error_email(error, synth):
    subject = f"ERROR Synth: {synth})"
    body = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    body += '\n' + str(error)
    log_email_target = 'lr2872@nyu.edu'
    send_email(subject, body, log_email_target)

def force_data_categorical_to_numeric(df, cat_columns=[]):
    # convert columns to categorical if they are not already
    for col in cat_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
    return df

def evaluate_model(model_filename, synth_class, epsilon, real_train_df, real_test_df, name_combo, omega, smallest_intersection):
    model = load_pickled_model(model_filename)
    try:
        synth_df = model.sample(real_train_df.shape[0])
    except Exception as e:
        error_email(str(e), 'ON GENERATION')
        print('ON GENERATION ' + model_filename)
        print(e)
        return pd.DataFrame()
    try:
        if name_combo == "vanilla":
            if synth_class.__name__ == 'GEMSynthesizer':
                synth_df = synth_df.loc[synth_df.apply(lambda row: (('SEX', row['SEX']), ('RAC1P', row['RAC1P'])) in smallest_intersection, axis=1)]
            else:
                synth_df = force_data_categorical_to_numeric(synth_df, cat_columns=synth_df.columns)
        performance_df = add_row_to_performance_df(pd.DataFrame(), synth_class, synth_df, epsilon, real_train_df, real_test_df, name_combo, omega)
        return performance_df
    except Exception as e:
        error_email(str(e), synth_class.__name__)
        print('Failed ' + model_filename)
        print(e)
        return pd.DataFrame()

def determine_limiting_synth(real_train_df, real_test_df, synthesizers, epsilon=0.1, strata_cols=['SEX','RAC1P'], seed=0):
    smallest_intersection = None
    for synth_class in synthesizers:
        for combination in combinations:
            name_combo = str("_".join(combination))
            model_filename = f"models/{synth_class.__name__}_epsilon_{epsilon}_{name_combo}_seed_{seed}.dill"
            synth_df = load_pickled_model(model_filename).sample(100000)
            intersection = set(create_subgroups_dict(real_train_df, ['SEX', 'RAC1P']).keys()).intersection(set(create_subgroups_dict(synth_df, ['SEX', 'RAC1P']).keys()))
            print("Intersection: ", len(intersection))
            if smallest_intersection is None or len(intersection) < len(smallest_intersection):
                smallest_intersection = intersection
    real_train_df = real_train_df.loc[real_train_df.apply(lambda row: (('SEX', row['SEX']), ('RAC1P', row['RAC1P'])) in smallest_intersection, axis=1)]
    real_test_df = real_test_df.loc[real_test_df.apply(lambda row: (('SEX', row['SEX']), ('RAC1P', row['RAC1P'])) in smallest_intersection, axis=1)]
    print("Smallest intersection: ", len(smallest_intersection))
    return real_train_df, real_test_df, smallest_intersection

def generate_performance_plots(real_train_df, real_test_df, combinations, synthesizers, epsilons):
    dataframe_cols = ['Synthesizer', 'Epsilon', 'Accuracy']
    for combination in list(combinations):
        strata_cols = list(combination)
        dataframe_cols.append(str(strata_cols))
    performance_df = pd.DataFrame(columns=dataframe_cols)

    # We want to only fit on rows that are represented by all synthesizers
    real_train_df, real_test_df, smallest_intersection = determine_limiting_synth(real_train_df, real_test_df, synthesizers)
    print(real_train_df.groupby(['RAC1P','SEX']).count())
    with ThreadPoolExecutor() as executor:
        futures = []
        for seed in [0,1,2,3,4]:
            for synth_class in synthesizers:
                for epsilon in epsilons:
                    model_filename = f"models/{synth_class.__name__}_epsilon_{epsilon}_seed_{seed}.dill"
                    futures.append(executor.submit(evaluate_model, model_filename, synth_class, epsilon, real_train_df, real_test_df, "vanilla", omega, smallest_intersection))

                    for combination in combinations:
                        name_combo = str("_".join(combination))
                        model_filename = f"models/{synth_class.__name__}_epsilon_{epsilon}_{name_combo}_seed_{seed}.dill"
                        futures.append(executor.submit(evaluate_model, model_filename, synth_class, epsilon, real_train_df, real_test_df, name_combo, omega, smallest_intersection))

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            result_df = future.result()
            performance_df = performance_df.append(result_df, ignore_index=True)
            if (i % 50) == 0:
                step_email(i)

    performance_df.to_pickle('aim_performance.pkl')

    return performance_df


# In[7]:


# Prepare the real dataset for evaluation
df_numeric = force_data_categorical_to_numeric(df, cat_columns=df.columns)
X_real = df_numeric.drop('ESR', axis=1)
y_real = df_numeric['ESR']
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
train_df_real = X_train_real.copy()
train_df_real['ESR'] = y_train_real
test_df_real = X_test_real.copy()
test_df_real['ESR'] = y_test_real

# Generate the performance plots
performance_df = generate_performance_plots(train_df_real, test_df_real, list(combinations), synthesizers, epsilons)


# In[9]:


step_email(0)


# In[ ]:




