def disparate_impact_1(X, subgroups_synth, f, M_f):
    """
    Calculate the disparate impact (DI) of a differentially private function M_f.
    
    Args:
    stratified_dataset (StratifiedDataset): A StratifiedDataset object containing the dataset and protected subgroups.
    f (Callable): A non-private function mapping from X* to real numbers.
    M_f (Callable): A differentially private function mapping from X* to real numbers.
    
    Returns:
    float: The disparate impact of M_f.
    """
    k = len(subgroups_synth)  # Number of protected subgroups
    total_diff = 0
    for i in range(k):
        for j in range(k):
            diff_i = np.abs(f(X) - M_f(subgroups_synth[i]))
            diff_j = np.abs(f(X) - M_f(subgroups_synth[j]))
            total_diff += np.abs(diff_i - diff_j)

    DI_M_f = total_diff / (k * k * f(X))

    return DI_M_f

def disparate_impact_over_synth_data(true_X, synth_df, di_func=disparate_impact_1, smoke_test=False):
    results_df = {}

    # Short circuit if smoke testing
    if smoke_test:
        for combination in list(combinations):
            strata_cols = list(combination)
            results_df[str(strata_cols)] = 0
        return results_df
    
    for combination in list(combinations):
        strata_cols = list(combination)
        stratified_synth_df = StratifiedDataset(synth_df, strata_cols, categorical_columns=synth_df.columns)
        subgroups = stratified_synth_df.get_strata_dfs()
        di_standard = di_func(true_X,
                              subgroups,
                              lambda x: np.mean(x['ESR'].astype(float)),
                              lambda x: np.mean(x['ESR'].astype(float)))
        results_df[str(strata_cols)] = di_standard
    return results_df
    
# di_standard = disparate_impact_over_synth_data(true_X=real_test_df, synth_df=synth_df)