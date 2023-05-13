import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import math


### Helper functions for the coinpress experiments  ###

# dirichlet draw
def dirichlet_draw(alpha, size):
    """
    Draw from a dirichlet distribution with parameter alpha.
    """
    return np.random.dirichlet(alpha, size=size)

# normalize numpy array method
def norm(v):
    return v/np.linalg.norm(v)

def safe_dir_draw(alphas, k, n):
    draw = np.random.dirichlet(alphas, size=k)[0]
    ret = n * draw
    index_max = np.argmax(ret)
    val_max = ret[index_max]
    for i, v in enumerate(ret):
        if v < 1:
            ret[i] = v + 1
            val_max = val_max - 1
    ret[index_max] = val_max
    return ret

def model(n=10000, mean=0.0, var=1.0, k=3, alpha=0.3, df=pd.DataFrame(), seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    all_alphas = np.full(int(k), alpha)
    dirichlet_draw = safe_dir_draw(all_alphas, k, n)
    group_sizes = np.round(dirichlet_draw).astype(int)
    # Ensure the sum of group_sizes is equal to n
    group_sizes[-1] = n - np.sum(group_sizes[:-1])
    
    means = np.random.normal(0, 1, size=k)
    X = np.empty(n)
    G = np.empty(n)
    
    start = 0
    for i in range(k):
        size = group_sizes[i]
        end = start + size
        X[start:end] = np.random.normal(means[i], var, size=size)
        G[start:end] = i
        start = end
    
    df_full = pd.DataFrame({'X': X, 'G': G})
    
    return df_full

def weights_public_sample(df, pub_sample_proportion=0.5):
    # assert that G and X are columns
    assert 'G' in df.columns
    assert 'X' in df.columns
    
    df = df.sample(frac=1, random_state=1)
    private_sample = df.sample(frac = pub_sample_proportion, random_state=1)
    public_sample = df.drop(private_sample.index)

    weights = {}
    for group in df.G.unique():
        group_frame = public_sample[public_sample.G == group]
        weights[group] = len(group_frame) / len(public_sample)

    return private_sample, weights

def weighted_mean(means, total, weights=None, use_weights=False, verbose=False):
    if verbose:
        return verbose_weighted_mean(means, total, weights, use_weights, verbose)
    else:
        return efficient_weighted_mean(means, total, weights, use_weights, verbose)

def verbose_weighted_mean(means, total, weights=None, use_weights=False, verbose=False):
    overall_mean = 0
    for mean, group_n, group in means:
        # add differentially private gaussian noise to n
        if verbose:
            print(f'Exact: {group_n}')
            print(f'Sampled: {(total * weights[group])}')
        if use_weights:
            overall_mean += mean * (total * weights[group])
        else:
            overall_mean += mean * group_n
    return overall_mean / total

def efficient_weighted_mean(means, total, weights=None, use_weights=False, verbose=False):
    return np.sum([(mean * (total * weights[group]) if use_weights else mean * group_n) for mean, group_n, group in means]) / total

def add_to_results_df(df, n, error, std, method, alpha, k):
    return df.append({
        'n': n,
        'error': error,
        'std': std,
        'method': method,
        'alpha': alpha,
        'k': k
    }, ignore_index=True)

### COINPRESS CODE ###

def L2(est): # assuming 0 vector is gt
    return np.linalg.norm(est)

def gaussian_tailbound(d,b):
    return ( d + 2*( d * math.log(1/b) )**0.5 + 2*math.log(1/b) )**0.5

def multivariate_mean_iterative(X, c, r, t, Ps):
    for i in range(t-1):
        c, r = multivariate_mean_step(X, c, r, Ps[i])
    c, r = multivariate_mean_step(X, c, r, Ps[t-1])
    return c

def multivariate_mean_step(X, c, r, p):
    n, d = X.shape

    ## Determine a good clipping threshold
    gamma = gaussian_tailbound(d,0.01)
    clip_thresh = min((r**2 + 2*r*3 + gamma**2)**0.5,r + gamma) #3 in place of sqrt(log(2/beta))
        
    ## Round each of X1,...,Xn to the nearest point in the ball B2(c,clip_thresh)
    x = X - c
    mag_x = np.linalg.norm(x, axis=1)
    outside_ball = (mag_x > clip_thresh)
    x_hat = (x.T / mag_x).T
    X[outside_ball] = c + (x_hat[outside_ball] * clip_thresh)
    
    ## Compute sensitivity
    delta = 2*clip_thresh/float(n)
    sd = delta/(2*p)**0.5
    
    ## Add noise calibrated to sensitivity
    Y = np.random.normal(0, sd, size=d)
    c = np.sum(X, axis=0)/float(n) + Y
    r = ( 1/float(n) + sd**2 )**0.5 * gaussian_tailbound(d,0.01)
    return c, r

########


### Plotting Functions ###

def plot_error_vs_k(results_df, alpha, n):
    df_filtered = results_df[(results_df['alpha'] == alpha) & (results_df['n'] == n)]
    df_filtered[['error', 'std', 'k']] = df_filtered[['error', 'std', 'k']].apply(pd.to_numeric)
    plt.figure(figsize=(10, 6))
    for method in df_filtered['method'].unique():
        method_data = df_filtered[df_filtered['method'] == method]
        plt.plot(method_data['k'], method_data['error'], '--', label=method)
        plt.fill_between(method_data['k'], method_data['error'] - method_data['std'], method_data['error'] + method_data['std'], alpha=0.2)
    plt.title(f'Mean Estimation Error vs. k (alpha={alpha}, n={n})')
    plt.legend()
    plt.show()

def plot_error_vs_alpha(results_df, k, n):
    df_filtered = results_df[(results_df['k'] == k) & (results_df['n'] == n)]
    df_filtered[['error', 'std', 'alpha']] = df_filtered[['error', 'std', 'alpha']].apply(pd.to_numeric)
    plt.figure(figsize=(10, 6))
    for method in df_filtered['method'].unique():
        method_data = df_filtered[df_filtered['method'] == method]
        plt.plot(method_data['alpha'], method_data['error'], '--', label=method)
        plt.fill_between(method_data['alpha'], method_data['error'] - method_data['std'], method_data['error'] + method_data['std'], alpha=0.2)
    plt.title(f'Mean Estimation Error vs. alpha (k={k}, n={n})')
    # log x axis
    plt.xscale('log')
    plt.legend()
    plt.show()

def plot_error_vs_n(results_df, alpha, k):
    df_filtered = results_df[(results_df['alpha'] == alpha) & (results_df['k'] == k)]
    df_filtered[['error', 'std', 'n']] = df_filtered[['error', 'std', 'n']].apply(pd.to_numeric)
    plt.figure(figsize=(10, 6))
    for method in df_filtered['method'].unique():
        method_data = df_filtered[df_filtered['method'] == method]
        plt.plot(method_data['n'], method_data['error'], '--', label=method)
        plt.fill_between(method_data['n'], method_data['error'] - method_data['std'], method_data['error'] + method_data['std'], alpha=0.2)
    plt.title(f'Mean Estimation Error vs. n (alpha={alpha}, k={k})')
    plt.legend()
    plt.show()

def plot_error_vs_alpha_and_n(results_df, k_list):
    filtered_results_df = results_df[results_df['k'].isin(k_list)]

    # Use catplot to create side-by-side bars for different methods within a FacetGrid
    g = sns.catplot(
        data=filtered_results_df,
        x='n',
        y='error',
        hue='method',
        col='k',
        row='alpha',
        kind='point',
        height=3,
        aspect=1.5,
        ci=None,
        dodge=True,
        order=sorted(filtered_results_df['n'].unique()),
    )
    g.set_axis_labels('n', 'Error')
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Error vs. alpha and n for different values of k')
    plt.show()

def toy_plot_for_mean_mixture(
    seed=5,
    n=10000,
    k=3,
    alpha=0.3,
    var=1.0,
):
    df = model(n=n, seed=seed, k=k, alpha=alpha, var=var)
    overall_mean = round(np.mean(df.X), 3)
    print(df.groupby('G').count())

    sns.kdeplot(df.X, legend=True)
    plt.axvline(overall_mean)
    plt.text(overall_mean + 0.1, 0, str(overall_mean),rotation=90)
    plt.xlabel('group')
    plt.ylabel('density')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(8,6))
    for group in df.G.unique():
        if group == 0:
            plt.axvline(overall_mean)
            plt.text(overall_mean + 0.1, 0, str(overall_mean),rotation=90)
        #     sns.kdeplot(df.X, label='all', legend=True)
        group_mean = round(np.mean(df.X[df.G == group]), 3)
        plt.axvline(group_mean)
        plt.text(group_mean + 0.1, 0, str(group_mean),rotation=90)
        sns.kdeplot(df.X[df.G == group],label=group, legend=True)

    # beautifying the labels
    plt.xlabel('group')
    plt.ylabel('density')
    plt.legend()
    plt.show()

########