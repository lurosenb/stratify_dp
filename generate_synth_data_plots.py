import pandas as pd
import numpy as np
from stratified_dataset import ParallelStratifiedSynthesizer
from snsynth.mst import MSTSynthesizer
from snsynth.aim import AIMSynthesizer
from gem_synthesizer import GEMSynthesizer
import dill
from data_utils import get_employment
import itertools
import os
from IPython.display import clear_output
from stratified_dataset import StratifiedDataset
import torch

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_pickled_model(filename, torch=False):
    with open(filename, "rb") as file:
        model = dill.load(file)
    return model

def evaluate_on_dataframes(train_df, test_df, target_col = 'ESR'):
    # Feature columns
    feature_cols = [col for col in train_df.columns if col != target_col]

    # Convert all columns to categorical
    for col in train_df.columns:
        train_df[col] = train_df[col].astype('category')
    
    for col in test_df.columns:
        test_df[col] = test_df[col].astype('category')

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

def get_subgroup_key(group, groupby_cols):
    key = []
    for col in groupby_cols:
        unique_values = group[col].unique()
        if len(unique_values) == 1:
            key.append((col, unique_values[0]))
        else:
            print(f"More than one unique value found for column '{col}' in the given group.")
            print(f"Unique values found: {unique_values}")
            print(f"Group:\n{group}")
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
            print(subgroups_synth)
            raise ValueError(f'Should not happen: {key} not in subgroups_synth')
            f_value_synth = f(X_prime)

        f_values_synth.append(f_value_synth)

    # Calculate the global f and M values
    f_global = f(X)
    f_synth_global = f(X_prime)

    # Compute the parity error
    beta = omega * (abs(f_global - f_synth_global) / f_global) + sum([(abs(t - s) / t) for t, s in zip(f_values_real, f_values_synth)])

    return beta

def mean_f(df):#, col='ESR'):
    return df.astype(float).mean().values

def add_row_to_performance_df(performance_df, synth_class, synth_df, epsilon, real_train_df, real_test_df, name_combo, omega = 0.2):
    accuracy = evaluate_on_dataframes(synth_df, real_test_df)
    row_dict = {
        'Synthesizer': synth_class.__name__ + "_" + name_combo,
        'Epsilon': epsilon,
        'Accuracy': accuracy,
    }
    for combination in combinations:
        strata_cols = list(combination)
        # TODO: Fix worst_accuracy 
        # worst_accuracy = evaluate_worst_accuracy(synth_df, real_test_df, strata_cols)
        # row_dict['Worst_Accuracy_' + str(strata_cols)] = worst_accuracy
        parity_error = parity_error_synth_data(real_train_df, synth_df, strata_cols, mean_f, omega)
        row_dict[str(strata_cols)] = parity_error
    performance_df = performance_df.append(row_dict, ignore_index=True)
    return performance_df

def force_data_categorical_to_numeric(df, cat_columns=[]):
    # convert columns to categorical if they are not already
    for col in cat_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
    return df

def evaluate_model(model_filename, synth_class, epsilon, real_train_df, real_test_df, name_combo, omega):
    model = load_pickled_model(model_filename)
    synth_df = model.sample(real_train_df.shape[0])
    # NOTE: double forcing here, sort by synth_class_name or add vanilla tag
    if name_combo == "vanilla":
        synth_df = force_data_categorical_to_numeric(synth_df, cat_columns=synth_df.columns)
    performance_df = add_row_to_performance_df(pd.DataFrame(), synth_class, synth_df, epsilon, real_train_df, real_test_df, name_combo, omega)
    return performance_df

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
    return real_train_df, real_test_df

def generate_performance_plots(real_train_df, real_test_df, combinations, synthesizers, epsilons):
    dataframe_cols = ['Synthesizer', 'Epsilon', 'Accuracy']
    for combination in list(combinations):
        strata_cols = list(combination)
        dataframe_cols.append(str(strata_cols))
    performance_df = pd.DataFrame(columns=dataframe_cols)

    # We want to only fit on rows that are represented by all synthesizers
    real_train_df, real_test_df = determine_limiting_synth(real_train_df, real_test_df, synthesizers)

    with ThreadPoolExecutor() as executor:
        futures = []

        for synth_class in synthesizers:
            for epsilon in epsilons:
                model_filename = f"models/{synth_class.__name__}_epsilon_{epsilon}_seed_{seed}.dill"
                futures.append(executor.submit(evaluate_model, model_filename, synth_class, epsilon, real_train_df, real_test_df, "vanilla", omega))

                for combination in combinations:
                    name_combo = str("_".join(combination))
                    model_filename = f"models/{synth_class.__name__}_epsilon_{epsilon}_{name_combo}_seed_{seed}.dill"
                    futures.append(executor.submit(evaluate_model, model_filename, synth_class, epsilon, real_train_df, real_test_df, name_combo, omega))

        for future in tqdm(as_completed(futures), total=len(futures)):
            result_df = future.result()
            performance_df = performance_df.append(result_df, ignore_index=True)

    performance_df.to_pickle('performance_df_gem.pkl')

    return performance_df



all_data, features, target, group = get_employment()

df = all_data.copy()

df = df.drop(columns=['CIT', 'MIG', 'DEAR', 'DEYE', 'NATIVITY', 'ANC'])

# List of column names you want to use
cols = ['SEX', 'RAC1P']
strata_cols = ['SEX', 'RAC1P']

synthesizers = [GEMSynthesizer] #[MSTSynthesizer, AIMSynthesizer]
epsilons = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
# Parity error param 
omega = 1/12 # (1/k is the default value in the paper)
seed = 0

# Generate all possible combinations of the given column names
combinations = []
for i in range(1, len(cols) + 1):
    combinations.extend(list(itertools.combinations(cols, i)))

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