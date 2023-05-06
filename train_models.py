
import pandas as pd
import numpy as np
from stratified_dataset import ParallelStratifiedSynthesizer
import dill
from data_utils import get_employment
import itertools
import os
from IPython.display import clear_output

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from snsynth.mst import MSTSynthesizer
from snsynth.aim import AIMSynthesizer
from gem_synthesizer import GEMSynthesizer

all_data, features, target, group = get_employment()

df = all_data.copy()
df = df.drop(columns=['CIT', 'MIG', 'DEAR', 'DEYE', 'NATIVITY', 'ANC'])

# List of column names you want to use
cols = ['SEX', 'RAC1P'] #, 'DIS', 'AGEP']

# Generate all possible combinations of the given column names
combinations = []
for i in range(1, len(cols) + 1):
    combinations.extend(list(itertools.combinations(cols, i)))

# Make models directory if one doesnt exist
if not os.path.exists("models"):
    os.mkdir("models")

# Make log text file
log_filename = "models/log.txt"
log_path = os.path.join(os.getcwd(), log_filename)
if not os.path.exists(log_path):
    with open(log_filename, "w") as file:
        file.write("")

def fit_vanilla_model(model, epsilon, df):
    model_filename = f"models/{model.__name__}_epsilon_{epsilon}.dill"
    
    model_path = os.path.join(os.getcwd(), model_filename)
    if os.path.exists(model_path):
        print(f"Model {model_filename} already exists. Skipping.")
        return
    
    m = model(epsilon=epsilon)
    m.fit(df)
    with open(model_filename, "wb") as file:
        dill.dump(m, file)

synthesizers = [MSTSynthesizer, AIMSynthesizer, GEMSynthesizer] #, MSTSynthesizer, AIMSynthesizer]

# Epsilon values to try
epsilons = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

for synth_class in synthesizers:
    for epsilon in epsilons:
        print(f"Training vanilla Synthesizer with {synth_class.__name__} and epsilon = {epsilon}")
        try:
            assert True == False
            fit_vanilla_model(MSTSynthesizer, epsilon, df)
        except:
            print(f"Failed to fit vanilla Synthesizer with {synth_class.__name__} with epsilon = {epsilon}")
            # Add to a log that we failed to fit this model
            with open(log_filename, "a") as file:
                file.write(f"Failed to fit vanilla Synthesizer with {synth_class.__name__} with epsilon = {epsilon}\n")

        for combination in list(combinations):
            strata_cols = list(combination)
            
            print(f"Training ParallelStratifiedSynthesizer with {synth_class.__name__} and epsilon = {epsilon}")

            name_combo = str("_".join(combination))
            model_filename = f"models/{synth_class.__name__}_epsilon_{epsilon}_{name_combo}.dill"
            
            # Check if the model file already exists, and if so, skip training and pickling
            model_path = os.path.join(os.getcwd(), model_filename)
            if os.path.exists(model_path):
                print(f"Model {model_filename} already exists. Skipping.")
                continue

            clear_output(wait=True)

            # Split the dataframe into train set
            train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
            try:
                assert True == False
                stratified_synth = ParallelStratifiedSynthesizer(synth_class, epsilon=epsilon)
                stratified_synth.fit(df, strata_cols=strata_cols, categorical_columns=df.columns)

                # Pickle the trained model
                with open(model_filename, "wb") as file:
                    dill.dump(stratified_synth, file)
                
                print(f"Model saved as {model_filename}")
            except:
                print(f"Failed to fit ParallelStratifiedSynthesizer with {synth_class.__name__} with epsilon = {epsilon} and strata_cols = {strata_cols}")
                # Add to a log that we failed to fit this model
                with open(log_filename, "a") as file:
                    file.write(f"Failed to fit ParallelStratifiedSynthesizer with {synth_class.__name__} with epsilon = {epsilon} and strata_cols = {strata_cols}\n")

print("All models trained and pickled.")